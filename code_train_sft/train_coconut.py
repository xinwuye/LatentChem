import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, TrainingArguments, TrainerCallback
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging
from datetime import datetime
import argparse
import wandb
import plotext as plt

# 导入我们的自定义组件
from model_new import Qwen3MoleculeLLM
from dataloader import load_data, COCONUT_TOKENS
from config import ModelConfig
from train_sft_stage2 import MultiModalDataCollator, MultiModalSFTTrainer, LoraTrainingMonitorCallback, TerminalPlotCallback, load_trained_components

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_coconut():
    parser = argparse.ArgumentParser(description="Coconut Training for Bio-LatentCOT")
    parser.add_argument("--data_path", type=str, default="/mnt/afs/L202500070/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot")
    parser.add_argument("--lora_path", type=str, default=None, help="Stage 2 LoRA weights (optional)")
    parser.add_argument("--projector_path", type=str, default=None, help="Stage 2 Projector weights (optional)")
    parser.add_argument("--output_dir", type=str, default="./outputs/stage3_coconut")
    parser.add_argument("--epochs_per_stage", type=int, default=3, help="How many epochs per latent stage")
    parser.add_argument("--max_latent_stage", type=int, default=3, help="Max number of CoT steps to latent-ize")
    parser.add_argument("--c_thought", type=int, default=2, help="Number of latent tokens per CoT step")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    
    args = parser.parse_args()

    # 1. 基础配置
    mol_config = {
        'num_queries': ModelConfig.NUM_QUERIES,
        'input_dim': ModelConfig.INPUT_DIM,
        'num_heads': ModelConfig.NUM_HEADS
    }
    
    # 当前的权重路径，初始为参数传入的路径
    current_lora_path = args.lora_path
    current_projector_path = args.projector_path

    # 2. 开启分阶段循环训练
    for stage in range(args.max_latent_stage + 1):
        logger.info(f"\n" + "🚀" * 30)
        logger.info(f"STARTING COCONUT STAGE {stage}")
        logger.info(f"Replace first {stage} steps with {stage * args.c_thought} latents")
        logger.info("🚀" * 30 + "\n")

        # 2.1 每一个 Stage 彻底重新初始化模型（为了彻底重置优化器和梯度状态）
        model = Qwen3MoleculeLLM(qwen_model_name=ModelConfig.DEFAULT_QWEN_PATH, mol_config=mol_config)
        tokenizer = model.tokenizer
        
        # 加载上一个 Stage 的权重（如果存在）
        if current_lora_path or current_projector_path:
            logger.info(f"Loading weights for stage {stage}...")
            model = load_trained_components(model, lora_weights_path=current_lora_path, projector_path=current_projector_path)
        
        # 确保 LoRA 已配置
        if not hasattr(model.model, 'peft_config') or model.model.peft_config is None:
            logger.info("Configuring LoRA from scratch...")
            # 首先冻结所有参数
            for param in model.parameters():
                param.requires_grad = False
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none",
            )
            model.model = get_peft_model(model.model, lora_config)
        
        # 确保投影器可训练
        for param in model.projector.parameters():
            param.requires_grad = True
        
        model.model.train()

        # 2.2 重新加载当前 Stage 的数据集
        train_dataset = load_data(
            args.data_path,
            is_coconut=True,
            scheduled_stage=stage,
            c_thought=args.c_thought,
            max_len=args.max_seq_length
        )

        # 2.3 配置当前 Stage 的输出目录
        stage_output_dir = os.path.join(args.output_dir, f"coconut_{stage}")
        
        training_args = SFTConfig(
            output_dir=stage_output_dir,
            num_train_epochs=args.epochs_per_stage, # 每个阶段练固定 Epoch
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            bf16=True,
            max_seq_length=args.max_seq_length,
            remove_unused_columns=False,
            logging_steps=10,
            save_strategy="no", # 🚨 不保存中间检查点，节省空间
            save_total_limit=1,
            gradient_checkpointing=True,
            # 🚨 修复 DDP 错误：使用非重入式 checkpoint 并允许查找未使用参数
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=True,
            report_to="wandb",
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            weight_decay=0.01,
        )

        # WandB 记录
        if wandb.run is not None:
            wandb.finish() # 结束上一个 stage 的 run
        
        wandb.init(
            project="qwen3-molecule-coconut",
            name=f"coconut-stage-{stage}-{datetime.now().strftime('%m%d-%H%M')}",
            mode="offline",
            config={**vars(args), "current_stage": stage}
        )

        data_collator = MultiModalDataCollator(tokenizer=tokenizer, model=model.model, padding=True)

        trainer = MultiModalSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[LoraTrainingMonitorCallback(), TerminalPlotCallback()],
        )

        # 执行当前阶段训练
        trainer.train()
        
        # 2.4 保存当前阶段结果，并更新下个阶段的加载路径
        trainer.save_model(stage_output_dir)
        current_lora_path = os.path.join(stage_output_dir, "lora_weights")
        current_projector_path = os.path.join(stage_output_dir, "projector.pt")
        
        # 手动保存 LoRA 和 Projector（Trainer 有时不会自动按我们的结构存）
        model.model.save_pretrained(current_lora_path)
        torch.save(model.projector.state_dict(), current_projector_path)
        
        logger.info(f"✅ Stage {stage} completed. Weights saved to {stage_output_dir}")
        
        # 显存清理
        del trainer, model, train_dataset
        torch.cuda.empty_cache()

    logger.info("🎉 All Coconut Stages completed!")

if __name__ == "__main__":
    train_coconut()

