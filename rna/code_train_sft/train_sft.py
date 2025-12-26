from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, TrainerCallback
from dataloader import load_data
from model_new import Qwen3MoleculeLLM
import torch
import os
from typing import Dict, List, Any
import logging
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 自定义回调函数，用于监控训练过程
class LoraTrainingMonitorCallback(TrainerCallback):
    """LoRA训练监控回调函数"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录训练日志"""
        if logs:
            if 'loss' in logs:
                logger.info(f"Step {state.global_step}: loss = {logs['loss']:.4f}")
            if 'learning_rate' in logs:
                logger.info(f"Step {state.global_step}: learning rate = {logs['learning_rate']:.6f}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时记录LoRA参数信息"""
        if 'model' in kwargs:
            model = kwargs['model']
            # 打印可训练参数信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"LoRA模型参数统计:")
            logger.info(f"  总参数: {total_params:,}")
            logger.info(f"  可训练参数: {trainable_params:,}")
            logger.info(f"  可训练比例: {100 * trainable_params / total_params:.2f}%")
            
            # 打印LoRA适配器信息
            if hasattr(model, 'peft_config'):
                for adapter_name, config in model.peft_config.items():
                    logger.info(f"  LoRA配置 - {adapter_name}:")
                    logger.info(f"    r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")

# 内存优化的collate_fn
def collate_fn(
    batch,
    smiles_len=10*5,           # 4 smiles + 2 special tokens
    pad_token_id=0,
    label_pad_id=-100,
):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        ids = x["input_ids"]
        mask = x["attention_mask"]
        lab = x["labels"]

        pad_len = max_len - len(ids)

        # text
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append(mask + [label_pad_id] * pad_len)

        # 🚨 关键：labels 对齐 logits
        labels.append(
            [label_pad_id] * smiles_len +   # smiles + special tokens
            lab  +                         # answer labels
            [label_pad_id] * pad_len         # padding
        )
        print("input_id_len",len(input_ids))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "smiles": [[x["smiles"].replace(".", "")] for x in batch],
    }


# LoRA SFT训练函数
def train_sft_lora(
    model_name="/zengdaojian/zhangjia/BioLatent/Qwen8B",
    data_path="/zengdaojian/zhangjia/BioLatent/ChemCotDataset/chemcotbench-cot",
    output_dir="./qwen3_mol_sft_lora_results",
    epochs=3,
    batch_size=32,  # LoRA可以使用稍大的批次
    lr=2e-4,
    max_seq_length=512,
):
    """
    使用LoRA进行SFT训练，显著减少内存使用
    """
    logger.info("=" * 60)
    logger.info("LoRA SFT Training for Qwen3MoleculeLLM")
    logger.info("=" * 60)
    
    # ============================
    # 1. 初始化基础模型
    # ============================
    logger.info("Initializing base model...")
    model = Qwen3MoleculeLLM(qwen_model_name=model_name)
    tokenizer = model.tokenizer
    
    # 确保pad_token设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # 冻结所有参数（只训练LoRA适配器）
    logger.info("Freezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    # ============================
    # 2. 配置LoRA
    # ============================
    logger.info("Configuring LoRA...")
    
    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA秩
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Qwen的模块
        bias="none",
    )
    
    # 将LoRA适配器添加到LLM部分
    logger.info("Adding LoRA adapters to LLM...")
    model.model = get_peft_model(model.model, lora_config)
    
    # 确保投影器可训练
    logger.info("Keeping projector trainable...")
    for param in model.projector.parameters():
        param.requires_grad = True
    
    # ============================
    # 3. 加载数据集
    # ============================
    logger.info(f"Loading dataset from {data_path}...")
    train_dataset = load_data(data_path)
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")
    
    # ============================
    # 4. 配置训练参数
    # ============================
    logger.info("Configuring training arguments...")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # 梯度累积
        learning_rate=lr,
        bf16=True,  # 使用bfloat16
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field=None,
        remove_unused_columns=False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        gradient_checkpointing=False,  # 启用梯度检查点
        max_grad_norm=0.3,  # 梯度裁剪
        warmup_ratio=0.1,
        report_to=[],  # 禁用wandb/tensorboard
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        optim="adamw_8bit",  # 8-bit优化器节省内存
    )
    
    # 打印训练配置
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Max sequence length: {max_seq_length}")
    logger.info(f"  LoRA r: {lora_config.r}")
    logger.info(f"  LoRA alpha: {lora_config.lora_alpha}")
    
    # ============================
    # 5. 初始化SFTTrainer
    # ============================
    logger.info("Initializing SFTTrainer...")
    
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        callbacks=[LoraTrainingMonitorCallback()],
    )
    
    # ============================
    # 6. 训练前验证
    # ============================
    logger.info("Testing forward pass...")
    try:
        # 测试一个样本
        if len(train_dataset) > 0:
            test_sample = [train_dataset[0]]
            test_batch = collate_fn(test_sample)
            
            # 移动到GPU
            if torch.cuda.is_available():
                device = torch.device("cuda")
                model = model.to(device)
                test_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in test_batch.items()}
            
            # 前向传播测试
            with torch.no_grad():
                outputs = model(
                    input_ids=test_batch["input_ids"],
                    attention_mask=test_batch["attention_mask"],
                    labels=test_batch["labels"],
                    smiles=test_batch["smiles"]
                )
            
            logger.info(f"Forward test successful!")
            logger.info(f"  Loss: {outputs.loss.item() if outputs.loss is not None else 'N/A'}")
            
            # 清理测试数据
            del test_sample, test_batch, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
    except Exception as e:
        logger.error(f"Forward test failed: {e}")
        raise
    
    # ============================
    # 7. 训练
    # ============================
    logger.info("Starting LoRA training...")
    
    try:
        trainer.train()
        
        logger.info("LoRA training completed!")
        
        # ============================
        # 8. 保存模型
        # ============================
        logger.info("Saving models...")
        
        # 保存完整的LoRA模型（包括基础模型）
        trainer.save_model(output_dir)
        
        # 单独保存LoRA适配器权重
        lora_weights_path = os.path.join(output_dir, "lora_weights")
        os.makedirs(lora_weights_path, exist_ok=True)
        model.model.save_pretrained(lora_weights_path)
        logger.info(f"LoRA weights saved to: {lora_weights_path}")
        
        # 保存投影器权重
        projector_path = os.path.join(output_dir, "projector.pt")
        torch.save(model.projector.state_dict(), projector_path)
        logger.info(f"Projector weights saved to: {projector_path}")
        
        # 保存分词器
        tokenizer.save_pretrained(output_dir)
        
        # ============================
        # 9. 合并LoRA权重（可选）
        # ============================
        logger.info("Merging LoRA weights with base model...")
        try:
            # 合并LoRA权重到基础模型
            merged_model = model.model.merge_and_unload()
            
            # 创建完整模型（包含合并的LoRA权重）
            merged_model_path = os.path.join(output_dir, "merged_model")
            os.makedirs(merged_model_path, exist_ok=True)
            
            # 保存合并后的模型
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            
            # 保存投影器配置
            torch.save(model.projector.state_dict(), os.path.join(merged_model_path, "projector.pt"))
            
            logger.info(f"Merged model saved to: {merged_model_path}")
            
        except Exception as e:
            logger.warning(f"Failed to merge LoRA weights: {e}")
            logger.warning("Using unmerged model for inference")
        
        return model
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory during LoRA training!")
        logger.error("Try reducing batch_size or max_seq_length")
        raise
    
    except Exception as e:
        logger.error(f"LoRA training failed: {e}")
        raise


# 加载LoRA模型进行推理 - 修复设备问题
def load_lora_model_for_inference(
    base_model_path="/zengdaojian/zhangjia/BioLatent/Qwen8B",
    lora_weights_path="./qwen3_mol_sft_lora_results/lora_weights",
    projector_path="./qwen3_mol_sft_lora_results/projector.pt",
    merge_lora=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    加载LoRA微调的模型进行推理 - 修复设备一致性
    """
    logger.info(f"Loading LoRA model for inference on {device}...")
    
    # 1. 加载基础模型
    model = Qwen3MoleculeLLM(qwen_model_name=base_model_path)
    tokenizer = model.tokenizer
    
    # 2. 确保pad_token设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 将模型移动到指定设备
    model = model.to(device)
    
    # 4. 加载LoRA权重
    from peft import PeftModel
    model.model = PeftModel.from_pretrained(model.model, lora_weights_path)
    
    # 5. 合并LoRA权重（可选，用于更快推理）
    if merge_lora:
        logger.info("Merging LoRA weights for faster inference...")
        model.model = model.model.merge_and_unload()
    
    # 6. 加载投影器权重并确保在正确设备上
    if os.path.exists(projector_path):
        # 加载时指定map_location
        projector_state_dict = torch.load(projector_path, map_location=device)
        model.projector.load_state_dict(projector_state_dict)
        # 确保投影器在正确设备上
        model.projector = model.projector.to(device)
        logger.info(f"Loaded projector weights to {device} from: {projector_path}")
    
    # 7. 确保模型各部分都在同一设备上
    model = model.to(device)
    
    # 8. 检查设备一致性
    model_devices = set()
    for name, param in model.named_parameters():
        model_devices.add(str(param.device))
    
    if len(model_devices) > 1:
        logger.warning(f"Model parameters are on multiple devices: {model_devices}")
        # 强制统一设备
        model = model.to(device)
    
    logger.info(f"Model loaded successfully on {device}")
    
    # 9. 设置为评估模式
    model.eval()
    
    return model, tokenizer


# 推理测试函数 - 修复设备问题
def test_lora_inference(
    model,
    tokenizer,
    test_smiles=[["CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl"]],
    test_prompts=["Modify the molecule CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl by adding a carboxyl."],
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    测试LoRA模型的推理能力 - 修复设备问题版本
    """
    logger.info(f"Testing LoRA model inference on {device}...")
    
    # 确保模型在正确设备上
    model.eval()
    model = model.to(device)
    
    results = []
    
    for smiles, prompt in zip(test_smiles, test_prompts):
        print(smiles)
        # 清理SMILES
        cleaned_smiles = [smile.replace(".", "").strip() for smile in smiles]
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Input SMILES: {cleaned_smiles}")
        logger.info(f"Input prompt: {prompt}")
        
        try:
            # 编码提示文本
            encodings = tokenizer(
                prompt,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            
            # 确保所有输入在相同设备上
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            
            # 打印设备信息以调试
            logger.info(f"Model device: {next(model.parameters()).device}")
            logger.info(f"Input IDs device: {input_ids.device}")
            logger.info(f"Attention mask device: {attention_mask.device}")
            
            # 生成回复 - 关键修复：确保smiles也在正确设备上处理
            with torch.no_grad():
                # 调用模型的generate方法
                generated_ids = model.generate(
                    smiles_list=[cleaned_smiles],  # SMILES列表
                    input_ids=input_ids,  # 文本输入
                    attention_mask=attention_mask,  # 注意力掩码
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
            
            # 解码结果
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # 提取生成的回答
            if generated_text.startswith(prompt):
                answer = generated_text[len(prompt):].strip()
            else:
                answer = generated_text
            
            logger.info(f"Generated response: {answer}")
            results.append({
                "smiles": cleaned_smiles,
                "prompt": prompt,
                "response": answer,
                "full_output": generated_text
            })
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # 尝试更简单的测试
            try:
                logger.info("Trying simpler forward pass...")
                
                # 创建一个简单的测试
                with torch.no_grad():
                    # 使用一个更简单的前向传播
                    test_inputs = {
                        "smiles": [cleaned_smiles],
                        "input_ids": torch.tensor([[tokenizer.bos_token_id]], device=device),
                        "attention_mask": torch.tensor([[1]], device=device),
                    }
                    
                    outputs = model(**test_inputs)
                    logger.info(f"Simple forward pass successful!")
                    
                results.append({
                    "smiles": cleaned_smiles,
                    "prompt": prompt,
                    "response": "[Model loaded but generation may have issues]",
                    "note": "Forward pass successful"
                })
                
            except Exception as e2:
                logger.error(f"Simple test also failed: {e2}")
                results.append({
                    "smiles": cleaned_smiles,
                    "prompt": prompt,
                    "error": str(e2)
                })
        
        logger.info(f"{'='*50}\n")
    
    return results


# 调试函数：检查模型设备状态


# 主函数
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA微调多模态分子-语言模型")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", help="运行模式")
    parser.add_argument("--model_path", type=str, default="./qwen3_8B_without_rnx_mol_sft_lora_results", help="模型路径")
    parser.add_argument("--data_path", type=str, default="/zengdaojian/zhangjia/BioLatent/ChemCotDataset/chemcotbench-cot", help="数据路径")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # 训练模式
        trained_model = train_sft_lora(
            data_path=args.data_path,
            output_dir=args.model_path,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            epochs=args.epochs,
        )
        
        # 训练完成后测试
        logger.info("Testing trained model...")
        test_lora_inference(
            trained_model,
            trained_model.tokenizer,
            test_smiles=[["CC(=O)OC1=CC=CC=C1C(=O)O"]],
            test_prompts=["Please describe the functional groups of this molecule."]
        )
    
    elif args.mode == "inference":
        # 推理模式
        model, tokenizer = load_lora_model_for_inference(
            lora_weights_path=os.path.join(args.model_path, "lora_weights"),
            projector_path=os.path.join(args.model_path, "projector.pt")
        )
        
        # 测试推理
        test_lora_inference(
            model,
            tokenizer,
            test_smiles=[["CC(=O)OC1=CC=CC=C1C(=O)O"]],
            test_prompts=["Modify the molecule CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl by adding a carboxyl."]
        )