import torch
import torch.nn as nn
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoTokenizer, 
    TrainerCallback, 
    DataCollatorForSeq2Seq
)
from dataloader import load_data, extract_fields, llm_tokenize, coconut_tokenize
from model_new import Qwen3MoleculeLLM
import os
import time
import json
import logging
import wandb
import plotext as plt
from tqdm import tqdm
from config import ModelConfig
from typing import Dict, List, Any, Optional
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 自定义回调函数，用于监控训练过程
class LoraTrainingMonitorCallback(TrainerCallback):
    """LoRA训练监控回调函数：仅负责打印关键信息，不重复记录 wandb"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """同时打印日志到控制台"""
        if logs:
            if 'loss' in logs:
                logger.info(f"Step {state.global_step}: loss = {logs['loss']:.4f}")
            if 'learning_rate' in logs:
                logger.info(f"Step {state.global_step}: lr = {logs['learning_rate']:.6f}")
    
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

# 终端绘图回调：用于在控制台实时显示 Loss 曲线
class TerminalPlotCallback(TrainerCallback):
    def __init__(self):
        self.steps = []
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            # 只有主进程负责绘图
            if state.is_world_process_zero:
                self.steps.append(state.global_step)
                self.losses.append(logs["loss"])
                
                # 保持最近的 100 个点以保证终端显示效果
                display_steps = self.steps[-100:]
                display_losses = self.losses[-100:]

                # 终端绘图逻辑
                plt.clf()
                plt.plot(display_steps, display_losses, marker="dot", color="red", label="SFT Loss")
                plt.title("Real-time Training Loss (Terminal)")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                
                # 设置合适终端大小的画布
                plt.plotsize(100, 25)
                plt.grid(True)
                plt.show()

# 工业级多模态数据整理器
class MultiModalDataCollator(DataCollatorForSeq2Seq):
    # ... (保持不变) ...
    def __call__(self, features):
        smiles = [f.pop("smiles") for f in features]
        batch = super().__call__(features)
        batch["smiles"] = smiles
        return batch

# 自定义 SFTTrainer 以支持多模态序列长度变化
class MultiModalSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 compute_loss。
        SFTTrainer 默认会在 compute_loss 中计算准确率指标，
        这要求 logits 和 labels 形状必须完全一致。
        在多模态场景下，序列会被拉长，因此我们回归标准 Trainer 的简单逻辑。
        
        注意：模型内部已经计算了正确的平均 Loss，这里直接返回即可。
        """
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
            return loss, outputs
        
        # 直接调用模型前向传播获取内部计算好的 Loss
        outputs = model(**inputs)
        
        # 模型已经返回正确的平均 Loss，直接使用
        return outputs.loss

# 加载已训练的模型组件
def load_trained_components(
    model,
    lora_weights_path=None,
    projector_path=None,
    device=None
):
    """
    加载已训练的组件到模型。
    取消所有 fallback 策略，加载失败即报错，确保训练基点正确。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载 LoRA 权重
    if lora_weights_path:
        if not os.path.exists(lora_weights_path):
            raise FileNotFoundError(f"LoRA weights path not found: {lora_weights_path}")
            
        logger.info(f"Loading LoRA weights from: {lora_weights_path}")
        # 强制使用 PeftModel 加载，不进行任何 try-except 容错
        model.model = PeftModel.from_pretrained(
            model.model, 
            lora_weights_path,
            is_trainable=True 
        )
        logger.info("Successfully loaded LoRA weights.")
    
    # 2. 加载投影器权重
    if projector_path:
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Projector weights path not found: {projector_path}")
            
        logger.info(f"Loading projector weights from: {projector_path}")
        # 强制加载状态字典
        projector_state_dict = torch.load(projector_path, map_location=device)
        model.projector.load_state_dict(projector_state_dict)
        logger.info("Successfully loaded projector weights.")
    
    return model

def train_sft_lora(
    model_name=None,
    data_path=None,
    output_dir=ModelConfig.DEFAULT_OUTPUT_DIR,
    epochs=3,
    batch_size=32,
    lr=2e-4,
    max_seq_length=8192,
    grad_accum=1,  # 新增：梯度累积步数
    resume_from_checkpoint=None,  # 新增：从检查点恢复训练
    lora_weights_path=None,  # 新增：预训练的LoRA权重路径
    projector_path=None,  # 新增：预训练的投影器权重路径
    wandb_project="qwen3-molecule-lora-sft",
    wandb_run_name=None,
    wandb_entity=None,
    mol_config=None,  # 新增：显式传入分子配置
    include_cot=True, # 新增：是否包含 CoT
):
    """
    使用LoRA进行SFT训练，支持从预训练权重继续训练
    """
    
    if model_name is None:
        model_name = ModelConfig.DEFAULT_QWEN_PATH
    if data_path is None:
        data_path = ModelConfig.DEFAULT_DATA_PATH
    
    # 如果没传 mol_config，则使用默认配置
    if mol_config is None:
        mol_config = {
            'num_queries': ModelConfig.NUM_QUERIES,
            'input_dim': ModelConfig.INPUT_DIM,
            'num_heads': ModelConfig.NUM_HEADS
        }
    
    logger.info("=" * 60)
    logger.info("LoRA SFT Training for Qwen3MoleculeLLM")
    logger.info(f"Resume from: {resume_from_checkpoint or 'scratch'}")
    logger.info("=" * 60)
    
    # ============================
    # 0. 初始化wandb
    # ============================
    # 如果未指定run_name，自动生成一个包含时间戳的名称
    if wandb_run_name is None:
        from datetime import datetime
        wandb_run_name = f"qwen3-lora-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 初始化wandb (使用 mode="offline" 替代环境变量设置)
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        entity=wandb_entity,
        mode="offline",
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
            "num_queries": mol_config['num_queries'],
            "include_cot": include_cot,
            "training_stage": "second_stage" if (resume_from_checkpoint or lora_weights_path) else "first_stage"
        }
    )
    
    # ============================
    # 1. 初始化基础模型
    # ============================
    logger.info("Initializing base model...")
    # 传入 mol_config
    model = Qwen3MoleculeLLM(qwen_model_name=model_name, mol_config=mol_config, device_map=None)
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
    # 无论是否加载了权重，我们都先确保基础模型是冻结的
    # 这是 PEFT 的标准实践：先全部冻结，再由 PEFT 开启特定层
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
        # 如果是加载的 PeftModel，PEFT 已经在 from_pretrained(..., is_trainable=True) 时处理好了梯度
        # 我们只需要确保它处于训练模式即可
        model.model.train()

    # 🚨 关键：只有 Projector 是我们需要手动处理的，因为它不在 LoRA 的管辖范围内
    logger.info("Ensuring projector is trainable...")
    for param in model.projector.parameters():
        param.requires_grad = True
    
    # ============================
    # 4. 加载数据集
    # ============================
    logger.info(f"Loading dataset from {data_path} (Include CoT: {include_cot}, Max Len: {max_seq_length})...")
    train_dataset = load_data(data_path, include_cot=include_cot, max_len=max_seq_length)
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
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        max_seq_length=max_seq_length,
        packing=False,
        dataset_text_field=None,
        remove_unused_columns=False,
        logging_steps=10,
        eval_strategy="no", # 不做划分，直接全部训练
        save_strategy="no", # 🚨 不保存中间检查点，节省空间
        save_total_limit=1,
        gradient_checkpointing=True, # 🚨 针对 8192 长度默认开启，防止 OOM
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        report_to="wandb", # 🚨 开启官方 wandb 支持
        dataloader_num_workers=2,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        resume_from_checkpoint=resume_from_checkpoint,
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
    # 6. 初始化 MultiModalSFTTrainer
    # ============================
    logger.info("Initializing MultiModalSFTTrainer...")
    
    # 使用工业级整理器
    data_collator = MultiModalDataCollator(
        tokenizer=tokenizer,
        model=model.model, # 传入基础 LLM 模型以获取 Padding 配置
        padding=True,
    )
    
    trainer = MultiModalSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LoraTrainingMonitorCallback(), TerminalPlotCallback()],
    )
    
    # ============================
    # 7. 训练前验证
    # ============================
    logger.info("Testing forward pass...")
    try:
        if len(train_dataset) > 0:
            test_sample = [train_dataset[0]]
            test_batch = data_collator(test_sample)
            
            # 获取模型主显卡
            device = next(model.parameters()).device
            
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
            # 记录 OOM 事件到 wandb
            wandb.log({"error/oom": 1})
            wandb.finish(exit_code=1)
        raise
    
    except Exception as e:
        logger.error(f"LoRA training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise

# 加载LoRA模型进行推理
def load_lora_model_for_inference(
    base_model_path=None,
    lora_weights_path="./qwen3_mol_sft_lora_results/lora_weights",
    projector_path="./qwen3_mol_sft_lora_results/projector.pt",
    merge_lora=True,
    device=None,
    mol_config=None  # 新增：分子配置
):
    """
    加载LoRA微调的模型进行推理 - 修复设备一致性
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    if base_model_path is None:
        base_model_path = ModelConfig.DEFAULT_QWEN_PATH
    
    if mol_config is None:
        mol_config = {
            'num_queries': ModelConfig.NUM_QUERIES,
            'input_dim': ModelConfig.INPUT_DIM,
            'num_heads': ModelConfig.NUM_HEADS
        }

    logger.info(f"Loading LoRA model for inference on {device}...")
    
    # 1. 加载基础模型，传入 mol_config
    model = Qwen3MoleculeLLM(qwen_model_name=base_model_path, mol_config=mol_config)
    tokenizer = model.tokenizer
    
    # 2. 确保pad_token设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 将模型移动到指定设备
    model = model.to(device)
    
    # 4. 加载LoRA权重
    from peft import PeftModel
    model.model = PeftModel.from_pretrained(model.model, lora_weights_path, device_map={"": str(device)})
    
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

# ------------------------------
# 以下为 inference / eval 相关替换部分
# ------------------------------

def load_test_data(test_data_path, max_len=None):
    """
    使用 dataloader.load_data 的 eval_mode 接口加载测试/验证数据（tokenized, labels=None）。

    如果 test_data_path 为目录：直接调用 load_data(path, eval_mode=True)
    如果 test_data_path 为单个文件：局部使用 load_dataset -> extract_fields(is_eval=True) -> llm_tokenize/coconut_tokenize 进行处理
    """

    if max_len is None:
        max_len = ModelConfig.MAX_TEXT_LEN

    logger.info(f"Loading test/eval data from: {test_data_path} (eval_mode=True)")

    # 情况1：目录（推荐）
    if os.path.isdir(test_data_path):
        # 直接使用 load_data 的 eval_mode
        if 'ChemCoTBench' in test_data_path:
            dataset = load_data(test_data_path, include_cot=False, is_coconut=False, eval_mode=True, exclude_tasks=['rcr', 'mechsel'], max_len=max_len)
            logger.info(f"Loaded tokenized eval dataset from dir: {len(dataset)} examples")
            return dataset
        elif 'ChemLLMBench' in test_data_path:
            dataset = load_data(test_data_path, include_cot=False, is_coconut=False, eval_mode=True, exclude_tasks=['name_prediction', 'property_prediction', 'yield_prediction'], max_len=max_len)
            logger.info(f"Loaded tokenized eval dataset from dir: {len(dataset)} examples")
            return dataset
    
def run_inference_on_test_data(
    model,
    tokenizer,
    test_data_path,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    save_results_path=None,
    batch_size=1,
    max_samples=None,
    tokenization_max_len=None,
    proc_index: int = 0,
    num_procs: int = 1,
    device = None
):
    """
    基于 load_data(..., eval_mode=True) 的推理流程。

    说明：
      - 该函数会调用 load_test_data(...) 获取已经 tokenized 的 eval 数据（labels 为 None）。
      - 目前实现以 sample-level (batch_size=1) 推理为主，保持逻辑简洁稳定。
    """
    logger.info(f"Running inference on test data from {test_data_path}")
    logger.info(f"Device: {device}, Max new tokens: {max_new_tokens}")

    # 加载并 tokenized 的 eval dataset
    dataset = load_test_data(test_data_path, max_len=tokenization_max_len)

    # 限制样本数量
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    # --------- 新增：按 proc_index / num_procs 分片 ------------
    if num_procs > 1:
        total = len(dataset)
        # 采用 strided slicing: proc_index, proc_index + num_procs, ...
        indices = list(range(proc_index, total, num_procs))
        dataset = dataset.select(indices)
        logger.info(f"Process {proc_index}/{num_procs} -> assigned {len(dataset)} samples (original total {total})")
    # ---------------------------------------------------------
    
    logger.info(f"Number of eval samples to run: {len(dataset)}")

    # 确保模型在正确设备上
    model.eval()
    model = model.to(device)

    results = []

    # 逐样本推理（保持原有逐个样本日志/错误捕获）
    for idx in tqdm(range(len(dataset))):
        item = dataset[idx]  # 已包含: input_ids (list[int]), attention_mask (list[int]), labels (None), smiles (list[str])

        try:
            smiles_list = item.get("smiles", []) or []
            # 清理 SMILES（去掉点）
            cleaned_smiles = [s.replace(".", "").strip() for s in smiles_list]

            # 构造 tensor inputs（单样本）
            input_ids = torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
            attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)

            # 生成
            with torch.no_grad():
                generated_ids = model.generate(
                    smiles_list=[cleaned_smiles],
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True if temperature > 0 else False,
                )

            # 兼容返回类型（tensor / list）
            if isinstance(generated_ids, torch.Tensor):
                gen0 = generated_ids[0]
            else:
                # 可能是 list of tensors
                gen0 = generated_ids[0]

            generated_text = tokenizer.decode(gen0, skip_special_tokens=True)
            # print(generated_text)

            result = {
                "sample_id": idx,
                "smiles": smiles_list,
                "result": generated_text.strip(),
                "task": item.get("task", None)
            }
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results.append({
                "sample_id": idx * num_procs + proc_index,
                "smiles": item.get("smiles", []),
                "result": None,
                "error": str(e),
                "task": item.get("task", None)
            })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}")
    logger.info(f"Inference completed on {len(results)} samples")

    # 保存结果（保持原有结构）
    if save_results_path:
        from datetime import datetime

        save_data = {
            "timestamp": datetime.now().isoformat(),
            "test_data_path": test_data_path,
            "model_info": {
                "device": str(device),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
            "generation_config": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
            "num_samples": len(results),
            "test_results": results
        }

        os.makedirs(os.path.dirname(save_results_path) if os.path.dirname(save_results_path) else ".", exist_ok=True)
        with open(save_results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {save_results_path}")

    return results

def load_vllm_model_for_inference(
    merged_model_path,
    projector_path,
    device,
    mol_config = None,
    tensor_parallel_size = 1,
    gpu_memory_utilization = 0.3
):
    if mol_config is None:
        mol_config = {
            'num_queries': ModelConfig.NUM_QUERIES,
            'input_dim': ModelConfig.INPUT_DIM,
            'num_heads': ModelConfig.NUM_HEADS
        }
    
    logger.info(f"Loading vLLM model for inference on {device}...")

    model = Qwen3MoleculeLLM(qwen_model_name=merged_model_path, mol_config=mol_config, vllm=True, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)
    tokenizer = model.tokenizer
    
    if os.path.exists(projector_path):
        # 加载时指定map_location
        projector_state_dict = torch.load(projector_path, map_location=device)
        model.projector.load_state_dict(projector_state_dict)
        logger.info(f"Loaded projector weights to {device} from: {projector_path}")
        model.projector = model.projector.to(device=device, dtype=torch.bfloat16)
    
    logger.info("mol_encoder device:")
    logger.info(next(model.mol_encoder.encoder.parameters()).device)
    
    logger.info(f"vLLM model lazy loaded for inference")

    return model, tokenizer

def run_inference_vllm(
    model,
    tokenizer,
    test_data_path,
    projector_device,
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    sample_count=1,
    save_results_path=None,
    max_samples=None,
    tokenization_max_len=None,
    embed_batch_size=32,
):
    logger.info(f"Running inference on test data from {test_data_path}")
    
    # 加载并 tokenized 的 eval dataset
    dataset = load_test_data(test_data_path, max_len=tokenization_max_len)

    # 限制样本数量
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        
    logger.info(f"Number of eval samples to run: {len(dataset)}")
    
    # -------------------------------------------------------
    # 1. 数据准备阶段 (Data Preparation)
    # -------------------------------------------------------
    # 目标：将 dataset 拆解为接口所需的三个大列表 (Lists)
    
    all_smiles_list: List[List[str]] = []
    all_input_ids: List[torch.Tensor] = []
    all_attention_mask: List[torch.Tensor] = []
    
    # 用于结果回填的元数据
    metadata_list: List[Dict[str, Any]] = []

    logger.info(f"Preparing inputs for {len(dataset)} samples...")

    for idx, item in enumerate(tqdm(dataset, desc="Preparing input data", unit="sample")):
        raw_smiles = item.get("smiles", []) or []
        cleaned_smiles = [s.replace(".", "").strip() for s in raw_smiles]

        # 确保转换为 1D Tensor (L,)，不需要 batch 维度
        inp = item["input_ids"]
        if isinstance(inp, list):
            inp_tensor = torch.tensor(inp, dtype=torch.long)
        elif isinstance(inp, torch.Tensor):
            inp_tensor = inp.clone().detach()
        else:
            # 防御性编程
            inp_tensor = torch.tensor(list(inp), dtype=torch.long)

        # 确保是 1D [L]
        if inp_tensor.dim() > 1:
            inp_tensor = inp_tensor.view(-1)

        # 1.3 处理 Attention Mask
        mask = item["attention_mask"]
        if isinstance(mask, list):
            mask_tensor = torch.tensor(mask, dtype=torch.long)
        elif isinstance(mask, torch.Tensor):
            mask_tensor = mask.clone().detach()
        else:
            mask_tensor = torch.tensor(list(mask), dtype=torch.long)

        # 确保是 1D [L]
        if mask_tensor.dim() > 1:
            mask_tensor = mask_tensor.view(-1)

        # 1.4 存入列表
        all_smiles_list.append(cleaned_smiles)
        all_input_ids.append(inp_tensor)
        all_attention_mask.append(mask_tensor)

        # 记录元数据，以便后续拼装结果
        metadata_list.append({
            "sample_id": idx, # 或 item.get('id')
            "original_smiles": raw_smiles,
            "task": item.get("task", None)
        })

    # -------------------------------------------------------
    # 2. 推理阶段 (Inference Phase)
    # -------------------------------------------------------
    # 直接调用你的接口，传入整包数据
    # 接口内部会处理 padding 和 mini-batching，我们无需操心
    
    logger.info(f"Starting batch inference via generate_vllm...")
    
    batch_outputs = model.generate_vllm(
        smiles_list=all_smiles_list,
        input_ids=all_input_ids,
        attention_mask=all_attention_mask,
        projector_device=projector_device,
        sample_count=sample_count,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        embed_batch_size=embed_batch_size,
        use_tqdm=True
    )

    # -------------------------------------------------------
    # 3. 结果重组 (Result Assembly)
    # -------------------------------------------------------

    results = []

    for meta, output_strs in tqdm(zip(metadata_list, batch_outputs),
                                  desc="Assembling results",
                                  unit="sample",
                                  total=len(metadata_list)):
        result_item = {
            "sample_id": meta["sample_id"],
            "smiles": meta["original_smiles"],
            "task": meta["task"]
        }
        if sample_count > 1:
            generated_texts = output_strs if output_strs else [""] * sample_count
            result_item['results'] = generated_texts
        else:
            generated_text = output_strs[0] if output_strs else ""
            result_item['results'] = generated_text
        print(generated_text)

        results.append(result_item)

    if save_results_path:
        from datetime import datetime

        save_data = {
            "timestamp": datetime.now().isoformat(),
            "test_data_path": test_data_path,
            # fill in vllm info later
            # "model_info": {
            #     "device": str(device),
            #     "total_parameters": sum(p.numel() for p in model.parameters()),
            #     "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            # },
            "generation_config": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "sample_count": sample_count
            },
            "num_samples": len(results),
            "test_results": results
        }

        os.makedirs(os.path.dirname(save_results_path) if os.path.dirname(save_results_path) else ".", exist_ok=True)
        with open(save_results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {save_results_path}")
    
    return results
    
# 主函数 - 修改以支持第二轮训练
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA微调多模态分子-语言模型")
    parser.add_argument("--mode", type=str, choices=["train", "inference", "vllm"], default="train", help="运行模式")
    parser.add_argument("--output_dir", type=str, default=ModelConfig.DEFAULT_OUTPUT_DIR, help="保存/加载模型的路径")
    parser.add_argument("--data_path", type=str, default=ModelConfig.DEFAULT_DATA_PATH, help="数据路径")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--max_seq_length", type=int, default=8192, help="最大序列长度")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--include_cot", type=lambda x: (str(x).lower() == 'true'), default=True, help="是否在 Response 中包含思维链 (CoT)")

    # 权重加载参数
    parser.add_argument("--lora_path", type=str, default=None, help="预训练 LoRA 权重路径")
    parser.add_argument("--projector_path", type=str, default=None, help="预训练投影器权重路径")
    parser.add_argument("--merged_model_path", type=str, default=None, help="预训练合并模型权重路径")

    # 分子模型超参数
    parser.add_argument("--num_queries", type=int, default=ModelConfig.NUM_QUERIES, help="投影器查询向量数量")
    parser.add_argument("--mol_input_dim", type=int, default=ModelConfig.INPUT_DIM, help="分子编码器输出维度")
    parser.add_argument("--mol_num_heads", type=int, default=ModelConfig.NUM_HEADS, help="投影器注意力头数")

    parser.add_argument("--resume_checkpoint", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--wandb_project", type=str, default="qwen3-molecule-lora-sft", help="wandb项目名称")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb运行名称")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb团队/实体名称")

    # 推理模式参数
    parser.add_argument("--test_data_path", type=str, default=None, help="测试数据路径（用于inference模式）")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="生成的最大token数（用于inference模式）")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度（用于inference模式）")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样参数（用于inference模式）")
    parser.add_argument("--max_test_samples", type=int, default=None, help="最大测试样本数，None表示全部测试（用于inference模式）")
    parser.add_argument("--inference_results_path", type=str, default=None, help="推理结果保存路径（用于inference模式）")
    parser.add_argument("--sample_count", type=int, default=1, help="每个样本生成的文本数量（用于inference模式）")
    parser.add_argument("--embed_batch_size", type=int, default=32, help="嵌入层批处理大小（用于vLLM推理模式）")

    # naive inference only
    parser.add_argument("--proc_index", type=int, default=0,
                    help="当前进程索引 (0-based)，用于样本分片")
    parser.add_argument("--num_procs", type=int, default=1,
                        help="并行进程总数（样本分片数）")
    parser.add_argument("--gpu", type=int, default=None,
                        help="显卡 id，优先于 proc_index (如果提供则使用此 GPU)")

    args = parser.parse_args()

    # 构建分子配置字典
    mol_config = {
        'num_queries': args.num_queries,
        'input_dim': args.mol_input_dim,
        'num_heads': args.mol_num_heads
    }

    if args.mode == "train":
        # 统一训练入口
        logger.info(f"Starting training mode...")
        trained_model = train_sft_lora(
            data_path=args.data_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            max_seq_length=args.max_seq_length,
            epochs=args.epochs,
            lora_weights_path=args.lora_path,
            projector_path=args.projector_path,
            resume_from_checkpoint=args.resume_checkpoint,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_entity=args.wandb_entity,
            mol_config=mol_config,
            include_cot=args.include_cot,
        )
        logger.info("Training completed!")

    elif args.mode == "inference":
        # 推理模式
        logger.info(f"Starting inference mode...")
        # 如果没有指定 lora_path，默认从 output_dir 寻找
        lora_path = args.lora_path or os.path.join(args.output_dir, "lora_weights")
        projector_path = args.projector_path or os.path.join(args.output_dir, "projector.pt")

        # 确定测试数据路径
        test_data_path = args.test_data_path or args.data_path
        if not test_data_path:
            raise ValueError("Please specify test data path using --test_data_path or --data_path")

        # 确定结果保存路径
        if args.inference_results_path:
            results_path = args.inference_results_path
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            results_path = os.path.join(args.output_dir, f"inference_results_{timestamp}.json")

        # 决定用于本进程的 device
        if torch.cuda.is_available():
            device = torch.device("cuda")   # 等价于 cuda:0（当前进程可见的第 0 张）
            torch.cuda.set_device(0)
        else:
            device = torch.device("cpu")

        model, tokenizer = load_lora_model_for_inference(
            base_model_path=None,
            lora_weights_path=lora_path,
            projector_path=projector_path,
            mol_config=mol_config,
            device=device,
            merge_lora=True
        )
        
        # 在测试数据上运行推理（使用基于 load_data(..., eval_mode=True) 的流程）
        run_inference_on_test_data(
            model=model,
            tokenizer=tokenizer,
            test_data_path=test_data_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            save_results_path=results_path,
            max_samples=args.max_test_samples,
            tokenization_max_len=min(args.max_seq_length, ModelConfig.MAX_TEXT_LEN),
            proc_index=args.proc_index,
            num_procs=args.num_procs,
            device=device
        )

        logger.info("Inference completed!")
    
    elif args.mode == "vllm":
        logger.info(f"Starting vLLM inference mode...")
        
        merged_model_path = args.merged_model_path or os.path.join(args.output_dir, "merged")
        projector_path = args.projector_path or os.path.join(args.output_dir, "projector.pt")
        
        # 确定测试数据路径
        test_data_path = args.test_data_path or args.data_path
        if not test_data_path:
            raise ValueError("Please specify test data path using --test_data_path or --data_path")

        # 确定结果保存路径
        if args.inference_results_path:
            results_path = args.inference_results_path
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            results_path = os.path.join(args.output_dir, f"inference_results_{timestamp}.json")
            
        model, tokenizer = load_vllm_model_for_inference(
            merged_model_path=merged_model_path,
            projector_path=projector_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            mol_config=mol_config,
            tensor_parallel_size=torch.cuda.device_count()
        )
        
        run_inference_vllm(
            model=model,
            tokenizer=tokenizer,
            test_data_path=test_data_path,
            save_results_path=results_path,
            projector_device="cuda" if torch.cuda.is_available() else "cpu",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_samples=args.max_test_samples,
            sample_count=args.sample_count,
            tokenization_max_len=min(args.max_seq_length, ModelConfig.MAX_TEXT_LEN),
            embed_batch_size=args.embed_batch_size,
        )
        
