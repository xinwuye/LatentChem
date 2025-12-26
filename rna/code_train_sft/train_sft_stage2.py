from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, TrainerCallback
from dataloader import load_data
from model_new import Qwen3MoleculeLLM
import torch
import os
from typing import Dict, List, Any
import logging
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
import wandb  # 新增：导入wandb

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 自定义回调函数，用于监控训练过程
class LoraTrainingMonitorCallback(TrainerCallback):
    """LoRA训练监控回调函数"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录训练日志到wandb"""
        if logs:
            # 记录到wandb
            if wandb.run is not None:
                wandb.log(logs, step=state.global_step)
            
            # 同时打印到控制台
            if 'loss' in logs:
                logger.info(f"Step {state.global_step}: loss = {logs['loss']:.4f}")
            if 'learning_rate' in logs:
                logger.info(f"Step {state.global_step}: learning rate = {logs['learning_rate']:.6f}")
            if 'grad_norm' in logs:
                logger.info(f"Step {state.global_step}: grad norm = {logs['grad_norm']:.4f}")
    
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
            
            # 记录到wandb
            if wandb.run is not None:
                wandb.config.update({
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "trainable_ratio": 100 * trainable_params / total_params
                })
            
            # 打印LoRA适配器信息
            if hasattr(model, 'peft_config'):
                for adapter_name, config in model.peft_config.items():
                    logger.info(f"  LoRA配置 - {adapter_name}:")
                    logger.info(f"    r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
                    
                    # 记录到wandb
                    if wandb.run is not None:
                        wandb.config.update({
                            f"lora_{adapter_name}_r": config.r,
                            f"lora_{adapter_name}_alpha": config.lora_alpha,
                            f"lora_{adapter_name}_dropout": config.lora_dropout
                        })
    
    def on_save(self, args, state, control, **kwargs):
        """保存检查点时记录"""
        if wandb.run is not None:
            wandb.log({"checkpoint_step": state.global_step})
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时记录"""
        if wandb.run is not None:
            wandb.log({"epoch": state.epoch})

# 内存优化的collate_fn
def collate_fn(
    batch,
    smiles_len=130*5,           # 4 smiles + 2 special tokens
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

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "smiles": [[x["smiles"].replace(".", "")] for x in batch],
    }

# 加载已训练的模型组件
def load_trained_components(
    model,
    lora_weights_path=None,
    projector_path=None,
    device=None
):
    """
    加载已训练的组件到模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载LoRA权重（如果存在）
    if lora_weights_path and os.path.exists(lora_weights_path):
        logger.info(f"Loading pre-trained LoRA weights from: {lora_weights_path}")
        
        # 检查是否是PeftModel（已包含LoRA权重）
        if hasattr(model.model, 'peft_config'):
            # 如果已经是PeftModel，使用from_pretrained加载适配器
            try:
                # 先加载配置
                peft_config = PeftConfig.from_pretrained(lora_weights_path)
                logger.info(f"Loaded LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")
                
                # 使用PeftModel加载权重
                model.model = PeftModel.from_pretrained(
                    model.model, 
                    lora_weights_path,
                    is_trainable=True  # 确保权重可训练
                )
                logger.info("Successfully loaded LoRA weights")
            except Exception as e:
                logger.warning(f"Failed to load LoRA weights as PeftModel: {e}")
                # 尝试其他加载方式
                try:
                    # 尝试直接加载状态字典
                    adapter_weights = torch.load(
                        os.path.join(lora_weights_path, "adapter_model.bin"),
                        map_location=device
                    )
                    model.model.load_state_dict(adapter_weights, strict=False)
                    logger.info("Loaded LoRA weights via state dict")
                except Exception as e2:
                    logger.warning(f"Failed to load LoRA weights via state dict: {e2}")
        else:
            # 如果不是PeftModel，尝试创建
            try:
                model.model = PeftModel.from_pretrained(
                    model.model,
                    lora_weights_path,
                    is_trainable=True
                )
                logger.info("Successfully loaded LoRA weights into base model")
            except Exception as e:
                logger.warning(f"Failed to load LoRA weights: {e}")
    
    # 加载投影器权重（如果存在）
    if projector_path and os.path.exists(projector_path):
        logger.info(f"Loading pre-trained projector weights from: {projector_path}")
        try:
            projector_state_dict = torch.load(projector_path, map_location=device)
            model.projector.load_state_dict(projector_state_dict)
            logger.info("Successfully loaded projector weights")
        except Exception as e:
            logger.warning(f"Failed to load projector weights: {e}")
    
    return model

# LoRA SFT训练函数 - 支持继续训练
def train_sft_lora(
    model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B",
    data_path="/zengdaojian/zhangjia/BioLatent/ChemCotDataset/chemcotbench-cot",
    output_dir="./qwen3_mol_sft_lora_results",
    epochs=3,
    batch_size=32,
    lr=2e-4,
    max_seq_length=8192,
    resume_from_checkpoint=None,  # 新增：从检查点恢复训练
    lora_weights_path=None,  # 新增：预训练的LoRA权重路径
    projector_path=None,  # 新增：预训练的投影器权重路径
    wandb_project="qwen3-molecule-lora-sft",
    wandb_run_name=None,
    wandb_entity=None,
):
    """
    使用LoRA进行SFT训练，支持从预训练权重继续训练
    """
    logger.info("=" * 60)
    logger.info("LoRA SFT Training for Qwen3MoleculeLLM")
    logger.info(f"Resume from: {resume_from_checkpoint or 'scratch'}")
    logger.info("=" * 60)
    
    # ============================
    # 0. 初始化wandb
    # ============================
    logger.info(f"Initializing wandb for experiment tracking...")
    try:
        # 如果未指定run_name，自动生成一个包含时间戳的名称
        if wandb_run_name is None:
            from datetime import datetime
            wandb_run_name = f"qwen3-lora-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # 初始化wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            entity=wandb_entity,
            config={
                "model_name": model_name,
                "data_path": data_path,
                "output_dir": output_dir,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "max_seq_length": max_seq_length,
                "training_strategy": "LoRA-SFT",
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "optimizer": "adamw_8bit",
                "mixed_precision": "bf16",
                "gradient_accumulation": 4,
                "resume_from_checkpoint": resume_from_checkpoint,
                "lora_weights_path": lora_weights_path,
                "projector_path": projector_path,
                "training_stage": "second_stage" if (resume_from_checkpoint or lora_weights_path) else "first_stage"
            }
        )
        
        logger.info(f"Wandb initialized: project={wandb_project}, run={wandb_run_name}")
        
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        logger.warning("Continuing without wandb...")
        wandb.init(mode="disabled")
    
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
    
    # ============================
    # 2. 加载预训练权重（如果提供）
    # ============================
    if lora_weights_path or projector_path:
        logger.info("Loading pre-trained components...")
        model = load_trained_components(
            model,
            lora_weights_path=lora_weights_path,
            projector_path=projector_path
        )
    
    # ============================
    # 3. 配置LoRA（如果还没有LoRA）
    # ============================
    if not hasattr(model.model, 'peft_config') or model.model.peft_config is None:
        logger.info("Configuring LoRA from scratch...")
        
        # 首先冻结所有参数
        logger.info("Freezing base model parameters...")
        for param in model.parameters():
            param.requires_grad = False
        
        # LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA秩
            lora_alpha=32,  # LoRA alpha
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        # 将LoRA适配器添加到LLM部分
        logger.info("Adding LoRA adapters to LLM...")
        model.model = get_peft_model(model.model, lora_config)
    else:
        logger.info("Using existing LoRA configuration")
        # 确保LoRA参数可训练
        for param in model.model.parameters():
            if hasattr(param, 'requires_grad'):
                param.requires_grad = True
    
    # 确保投影器可训练
    logger.info("Keeping projector trainable...")
    for param in model.projector.parameters():
        param.requires_grad = True
    
    # ============================
    # 4. 加载数据集
    # ============================
    logger.info(f"Loading dataset from {data_path}...")
    train_dataset = load_data(data_path)
    logger.info(f"Dataset loaded: {len(train_dataset)} samples")
    
    # 记录数据集信息到wandb
    if wandb.run is not None:
        wandb.config.update({
            "dataset_size": len(train_dataset),
            "dataset_path": data_path,
        })
    
    # ============================
    # 5. 配置训练参数
    # ============================
    logger.info("Configuring training arguments...")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        bf16=True,
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field=None,
        remove_unused_columns=False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        gradient_checkpointing=False,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        report_to=["wandb"],
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        resume_from_checkpoint=resume_from_checkpoint,  # 支持从检查点恢复
    )
    
    # 打印训练配置
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {lr}")
    logger.info(f"  Max sequence length: {max_seq_length}")
    logger.info(f"  Resume from: {resume_from_checkpoint or 'None'}")
    logger.info(f"  Pre-trained LoRA: {lora_weights_path or 'None'}")
    logger.info(f"  Pre-trained projector: {projector_path or 'None'}")
    
    # ============================
    # 6. 初始化SFTTrainer
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
    # 7. 训练前验证
    # ============================
    logger.info("Testing forward pass...")
    try:
        if len(train_dataset) > 0:
            test_sample = [train_dataset[0]]
            test_batch = collate_fn(test_sample)
            
            if torch.cuda.is_available():
                device = torch.device("cuda")
                model = model.to(device)
                test_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in test_batch.items()}
            
            with torch.no_grad():
                outputs = model(
                    input_ids=test_batch["input_ids"],
                    attention_mask=test_batch["attention_mask"],
                    labels=test_batch["labels"],
                    smiles=test_batch["smiles"]
                )
            
            logger.info(f"Forward test successful!")
            initial_loss = outputs.loss.item() if outputs.loss is not None else float('nan')
            logger.info(f"  Initial Loss: {initial_loss}")
            
            if wandb.run is not None:
                wandb.log({"initial_loss": initial_loss})
            
            del test_sample, test_batch, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
    except Exception as e:
        logger.error(f"Forward test failed: {e}")
        raise
    
    # ============================
    # 8. 训练
    # ============================
    logger.info("Starting LoRA training...")
    
    try:
        import time
        start_time = time.time()
        
        # 开始训练
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        training_time = time.time() - start_time
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info("LoRA training completed!")
        logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        if wandb.run is not None:
            wandb.log({"total_training_time_hours": training_time / 3600})
            wandb.config.update({"total_training_time_seconds": training_time})
        
        # ============================
        # 9. 保存模型
        # ============================
        logger.info("Saving models...")
        
        # 保存完整的模型
        trainer.save_model(output_dir)
        
        # 单独保存LoRA适配器权重
        lora_weights_path_save = os.path.join(output_dir, "lora_weights")
        os.makedirs(lora_weights_path_save, exist_ok=True)
        model.model.save_pretrained(lora_weights_path_save)
        logger.info(f"LoRA weights saved to: {lora_weights_path_save}")
        
        # 保存投影器权重
        projector_path_save = os.path.join(output_dir, "projector.pt")
        torch.save(model.projector.state_dict(), projector_path_save)
        logger.info(f"Projector weights saved to: {projector_path_save}")
        
        # 保存分词器
        tokenizer.save_pretrained(output_dir)
        
        # 保存训练配置
        config_save_path = os.path.join(output_dir, "training_config.json")
        import json
        config_dict = {
            "model_name": model_name,
            "data_path": data_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_seq_length": max_seq_length,
            "resume_from_checkpoint": resume_from_checkpoint,
            "pretrained_lora": lora_weights_path,
            "pretrained_projector": projector_path,
            "training_time_seconds": training_time,
            "training_stage": "second_stage" if (resume_from_checkpoint or lora_weights_path) else "first_stage"
        }
        with open(config_save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"training-config-{wandb_run_name}",
                type="config",
                description="Training configuration file"
            )
            artifact.add_file(config_save_path)
            wandb.log_artifact(artifact)
        
        # ============================
        # 10. 合并LoRA权重（可选）
        # ============================
        logger.info("Merging LoRA weights with base model...")
        try:
            merged_model = model.model.merge_and_unload()
            
            merged_model_path = os.path.join(output_dir, "merged_model")
            os.makedirs(merged_model_path, exist_ok=True)
            
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            
            torch.save(model.projector.state_dict(), os.path.join(merged_model_path, "projector.pt"))
            
            logger.info(f"Merged model saved to: {merged_model_path}")
            
            if wandb.run is not None:
                lora_artifact = wandb.Artifact(
                    name=f"lora-weights-{wandb_run_name}",
                    type="model",
                    description="LoRA adapter weights"
                )
                lora_artifact.add_dir(lora_weights_path_save)
                wandb.log_artifact(lora_artifact)
                
                model_artifact = wandb.Artifact(
                    name=f"full-model-{wandb_run_name}",
                    type="model",
                    description="Full model with merged LoRA weights"
                )
                model_artifact.add_dir(merged_model_path)
                wandb.log_artifact(model_artifact)
            
        except Exception as e:
            logger.warning(f"Failed to merge LoRA weights: {e}")
            logger.warning("Using unmerged model for inference")
        
        # 完成wandb运行
        if wandb.run is not None:
            wandb.finish()
        
        return model
        
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory during LoRA training!")
        logger.error("Try reducing batch_size or max_seq_length")
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise
    
    except Exception as e:
        logger.error(f"LoRA training failed: {e}")
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise

# 加载LoRA模型进行推理
def load_lora_model_for_inference(
    base_model_path="/zengdaojian/zhangjia/BioLatent/Qwen4B",
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
            print(generated_text)
            
            # 提取生成的回答
            # if generated_text.startswith(prompt):
            #     answer = generated_text[len(prompt):].strip()
            # else:
            #     answer = generated_text
            answer=generated_text[len(prompt)+66*5:].strip()
            
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


# 批量测试函数
def batch_test_lora_inference(
    model,
    tokenizer,
    test_cases=None,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_results_path=None
):
    """
    批量测试LoRA模型的推理能力
    
    Args:
        model: 加载的模型
        tokenizer: 分词器
        test_cases: 测试用例列表，每个元素是(smiles_list, prompt)元组
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_p: top-p采样参数
        device: 设备
        save_results_path: 保存结果的文件路径
    
    Returns:
        results: 推理结果列表
    """
    if test_cases is None:
        # 默认测试用例
        test_cases = [
            (["CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl"], 
             "Modify the molecule CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl by adding a carboxyl."),
            (["CCO"], 
             "Describe the properties of ethanol (CCO)."),
            (["CC(C)Cc1ccc(cc1)C(C)C(=O)O"], 
             "What is the IUPAC name of this molecule?")
        ]
    
    # 解包测试用例
    test_smiles = [case[0] for case in test_cases]
    test_prompts = [case[1] for case in test_cases]
    
    # 执行测试
    results = test_lora_inference(
        model=model,
        tokenizer=tokenizer,
        test_smiles=test_smiles,
        test_prompts=test_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
        verbose=True
    )
    
    # 保存结果
    if save_results_path:
        import json
        from datetime import datetime
        
        # 准备保存的数据
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "device": str(device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            "generation_config": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            "test_results": results
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        
        # 保存到JSON文件
        with open(save_results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {save_results_path}")
    
    return results


# 主函数 - 修改以支持第二轮训练
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LoRA微调多模态分子-语言模型")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "continue"], default="train", help="运行模式")
    parser.add_argument("--model_path", type=str, default="./qwen3_4B_without_128_cot_new_rnx_mol_sft_lora_results_stage2", help="模型路径")
    parser.add_argument("--data_path", type=str, default="/zengdaojian/zhangjia/BioLatent/ChemCotDataset/chemcotbench-cot", help="数据路径")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--wandb_project", type=str, default="qwen3-molecule-lora-sft", help="wandb项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb运行名称")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb团队/实体名称")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # 第一轮训练（从头开始）
        trained_model = train_sft_lora(
            data_path=args.data_path,
            output_dir=args.model_path,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            epochs=args.epochs,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_entity=args.wandb_entity,
        )
        
        # 训练完成后测试
        logger.info("Testing trained model...")
        # test_lora_inference函数保持不变
        # ...
    
    elif args.mode == "continue":
        # 第二轮训练（从第一轮保存的参数继续）
        logger.info("Starting second stage training with pre-trained weights...")
        
        # 构建预训练权重路径
        lora_weights_path = os.path.join(args.model_path, "lora_weights")
        projector_path = os.path.join(args.model_path, "projector.pt")
        
        # 检查权重文件是否存在
        if not os.path.exists(lora_weights_path):
            logger.warning(f"LoRA weights not found at: {lora_weights_path}")
            logger.warning("Falling back to training from scratch...")
            lora_weights_path = None
        
        if not os.path.exists(projector_path):
            logger.warning(f"Projector weights not found at: {projector_path}")
            projector_path = None
        
        # 进行第二轮训练
        trained_model = train_sft_lora(
            data_path=args.data_path,
            output_dir=args.model_path + "_stage2",  # 使用不同的输出目录
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            epochs=args.epochs,
            lora_weights_path=lora_weights_path,
            projector_path=projector_path,
            resume_from_checkpoint=args.resume_checkpoint,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name + "-stage2" if args.wandb_run_name else "qwen3-lora-sft-stage2",
            wandb_entity=args.wandb_entity,
        )
        
        logger.info("Second stage training completed!")
    
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
            test_smiles=[["CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl"]],
            test_prompts=["Modify the molecule CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl by adding a carboxyl."]
        )