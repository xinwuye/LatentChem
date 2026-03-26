import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    TrainerCallback, 
    DataCollatorForSeq2Seq
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging
from datetime import datetime
import argparse
import inspect
import sys
import wandb
import plotext as plt

# 导入我们的自定义组件
from model_stage3 import Qwen3MoleculeLLM
from dataloader import load_data, set_tokenizer_model_path
from config import ModelConfig
# from train_sft_stage2 import MultiModalDataCollator, MultiModalSFTTrainer, LoraTrainingMonitorCallback, TerminalPlotCallback
import torch.nn.functional as F
import random
from transformers import set_seed as hf_set_seed

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_qwen_model_path(qwen_size: str) -> str:
    return ModelConfig.require_qwen_path(qwen_size)


def _arg_was_explicitly_provided(argv: list[str], flag: str) -> bool:
    if not flag.startswith("--"):
        raise ValueError(f"Expected a long CLI flag like '--seed', got: {flag}")
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)

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
        cot_len = [f.pop("cot_len") for f in features] if features and ("cot_len" in features[0]) else None
        batch = super().__call__(features)
        batch["smiles"] = smiles
        if cot_len is not None:
            batch["cot_len"] = torch.tensor(cot_len, dtype=torch.long)
        return batch

# 自定义 SFTTrainer 以支持多模态序列长度变化
class MultiModalSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        cf_lambda: float = 0.0,
        cf_margin: float = 0.5,
        cf_prob: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cf_lambda = float(cf_lambda)
        self.cf_margin = float(cf_margin)
        self.cf_prob = float(cf_prob)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 compute_loss。
        SFTTrainer 默认会在 compute_loss 中计算准确率指标，
        这要求 logits 和 labels 形状必须完全一致。
        在多模态场景下，序列会被拉长，因此我们回归标准 Trainer 的简单逻辑。
        
        注意：模型内部已经计算了正确的平均 Loss，这里直接返回即可。
        """
        def _out_get(outputs, key, default=None):
            if isinstance(outputs, dict):
                return outputs.get(key, default)
            return getattr(outputs, key, default)

        def _is_world_zero():
            fn = getattr(self, "is_world_process_zero", None)
            return fn() if callable(fn) else bool(fn)

        # --- Clean forward pass (no perturbation) ---
        outputs_pos = model(**inputs, do_perturb=False)
        loss_pos = outputs_pos.loss
        ce_loss_pos = _out_get(outputs_pos, "ce_loss", None)
        if ce_loss_pos is None:
            ce_loss_pos = loss_pos
        bio_latent_active_pos = bool(_out_get(outputs_pos, "bio_latent_active", False))
        bio_latent_loss_pos = _out_get(outputs_pos, "bio_latent_loss", None)
        bio_latent_loss_scaled_pos = _out_get(outputs_pos, "bio_latent_loss_scaled", None)
        task_latent_active_pos = bool(_out_get(outputs_pos, "task_latent_active", False))
        task_latent_loss_pos = _out_get(outputs_pos, "task_latent_loss", None)
        task_latent_loss_scaled_pos = _out_get(outputs_pos, "task_latent_loss_scaled", None)

        # Optionally enable counterfactual loss (paired pass)
        do_cf = (
            (self.cf_lambda is not None and self.cf_lambda > 0.0)
            and (random.random() < self.cf_prob)
        )

        if not do_cf:
            if _is_world_zero():
                step = int(getattr(self.state, "global_step", 0) or 0)
                log_every = int(getattr(getattr(self, "args", None), "logging_steps", 10) or 10)

                if wandb.run is not None:
                    log_dict = {
                        "loss_pos": loss_pos.detach().float().item(),
                        "ce_loss_pos": ce_loss_pos.detach().float().item(),
                        "bio_latent_active": 1.0 if bio_latent_active_pos else 0.0,
                        "task_latent_active": 1.0 if task_latent_active_pos else 0.0,
                    }
                    if bio_latent_loss_pos is not None:
                        log_dict["bio_latent_loss_pos"] = bio_latent_loss_pos.detach().float().item()
                    if bio_latent_loss_scaled_pos is not None:
                        log_dict["bio_latent_loss_scaled_pos"] = bio_latent_loss_scaled_pos.detach().float().item()
                    if task_latent_loss_pos is not None:
                        log_dict["task_latent_loss_pos"] = task_latent_loss_pos.detach().float().item()
                    if task_latent_loss_scaled_pos is not None:
                        log_dict["task_latent_loss_scaled_pos"] = task_latent_loss_scaled_pos.detach().float().item()
                    wandb.log(log_dict)

                if log_every > 0 and (step % log_every == 0):
                    if getattr(self, "_last_subloss_print_step", None) != step:
                        self._last_subloss_print_step = step
                        msg = f"Step {step}: ce_loss={ce_loss_pos.detach().float().item():.4f}"
                        if bio_latent_active_pos and bio_latent_loss_pos is not None:
                            msg += f", bio_latent_loss={bio_latent_loss_pos.detach().float().item():.4f}"
                        if task_latent_active_pos and task_latent_loss_pos is not None:
                            msg += f", task_latent_loss={task_latent_loss_pos.detach().float().item():.4f}"
                        logger.info(msg)
            return (loss_pos, outputs_pos) if return_outputs else loss_pos

        # --- Corrupted forward pass ---
        # 🚨 DDP FIX: If the model is wrapped in DDP, calling it twice triggers "marked ready twice" error.
        # We call the underlying module for the second pass to avoid this.
        # Gradients from both passes will be summed in the .grad fields and synced by DDP's first-pass hook.
        unwrapped_model = model.module if hasattr(model, "module") else model
        outputs_cf = unwrapped_model(**inputs, do_perturb=True)
        loss_cf = outputs_cf.loss
        ce_loss_cf = _out_get(outputs_cf, "ce_loss", None)
        if ce_loss_cf is None:
            ce_loss_cf = loss_cf
        bio_latent_active_cf = bool(_out_get(outputs_cf, "bio_latent_active", False))
        bio_latent_loss_cf = _out_get(outputs_cf, "bio_latent_loss", None)
        bio_latent_loss_scaled_cf = _out_get(outputs_cf, "bio_latent_loss_scaled", None)
        task_latent_active_cf = bool(_out_get(outputs_cf, "task_latent_active", False))
        task_latent_loss_cf = _out_get(outputs_cf, "task_latent_loss", None)
        task_latent_loss_scaled_cf = _out_get(outputs_cf, "task_latent_loss_scaled", None)

        # Hinge on CE gap: enforce L_cf - L_pos >= margin
        gap = loss_cf - loss_pos
        loss_cf_term = F.relu(self.cf_margin - gap)
        loss_total = loss_pos + (self.cf_lambda * loss_cf_term)

        if _is_world_zero():
            step = int(getattr(self.state, "global_step", 0) or 0)
            log_every = int(getattr(getattr(self, "args", None), "logging_steps", 10) or 10)

            if wandb.run is not None:
                log_dict = {
                    "loss_pos": loss_pos.detach().float().item(),
                    "loss_cf": loss_cf.detach().float().item(),
                    "cf_gap": gap.detach().float().item(),
                    "loss_cf_term": loss_cf_term.detach().float().item(),
                    "loss_cf_scaled": (self.cf_lambda * loss_cf_term).detach().float().item(),
                    "loss_total": loss_total.detach().float().item(),
                    "ce_loss_pos": ce_loss_pos.detach().float().item(),
                    "ce_loss_cf": ce_loss_cf.detach().float().item(),
                    "bio_latent_active": 1.0 if (bio_latent_active_pos or bio_latent_active_cf) else 0.0,
                    "task_latent_active": 1.0 if (task_latent_active_pos or task_latent_active_cf) else 0.0,
                }
                if bio_latent_loss_pos is not None:
                    log_dict["bio_latent_loss_pos"] = bio_latent_loss_pos.detach().float().item()
                if bio_latent_loss_scaled_pos is not None:
                    log_dict["bio_latent_loss_scaled_pos"] = bio_latent_loss_scaled_pos.detach().float().item()
                if bio_latent_loss_cf is not None:
                    log_dict["bio_latent_loss_cf"] = bio_latent_loss_cf.detach().float().item()
                if bio_latent_loss_scaled_cf is not None:
                    log_dict["bio_latent_loss_scaled_cf"] = bio_latent_loss_scaled_cf.detach().float().item()
                if task_latent_loss_pos is not None:
                    log_dict["task_latent_loss_pos"] = task_latent_loss_pos.detach().float().item()
                if task_latent_loss_scaled_pos is not None:
                    log_dict["task_latent_loss_scaled_pos"] = task_latent_loss_scaled_pos.detach().float().item()
                if task_latent_loss_cf is not None:
                    log_dict["task_latent_loss_cf"] = task_latent_loss_cf.detach().float().item()
                if task_latent_loss_scaled_cf is not None:
                    log_dict["task_latent_loss_scaled_cf"] = task_latent_loss_scaled_cf.detach().float().item()
                wandb.log(log_dict)

            if log_every > 0 and (step % log_every == 0):
                if getattr(self, "_last_subloss_print_step", None) != step:
                    self._last_subloss_print_step = step
                    msg = (
                        f"Step {step}: ce_pos={ce_loss_pos.detach().float().item():.4f}, "
                        f"ce_cf={ce_loss_cf.detach().float().item():.4f}, "
                        f"cf_term={loss_cf_term.detach().float().item():.4f}"
                    )
                    if bio_latent_active_pos and bio_latent_loss_pos is not None:
                        msg += f", bio_latent_loss_pos={bio_latent_loss_pos.detach().float().item():.4f}"
                    if task_latent_active_pos and task_latent_loss_pos is not None:
                        msg += f", task_latent_loss_pos={task_latent_loss_pos.detach().float().item():.4f}"
                    logger.info(msg)

        return (loss_total, outputs_pos) if return_outputs else loss_total


def load_trained_components_stage3(model, lora_weights_path=None, mm_projector_path=None):
    """
    Unified loader for Stage 3: loads LoRA weights and combined Projector + Bio Updater weights.
    Assumes unified multi-modal checkpoint format.
    """
    def _ensure_exists(path: str, *, kind: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{kind} path does not exist: {path}")

    def _ensure_is_dir(path: str, *, kind: str) -> None:
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{kind} path must be a directory: {path}")

    def _ensure_is_file(path: str, *, kind: str) -> None:
        if not os.path.isfile(path):
            raise IsADirectoryError(f"{kind} path must be a file: {path}")

    def _require_ckpt_key(ckpt: dict, key: str, *, path: str) -> None:
        if key not in ckpt:
            found = ", ".join(sorted(map(str, ckpt.keys())))
            raise KeyError(f"Checkpoint {path} is missing key '{key}'. Found keys: [{found}]")

    # 1. 加载 LoRA 权重
    if lora_weights_path:
        _ensure_exists(lora_weights_path, kind="LoRA")
        _ensure_is_dir(lora_weights_path, kind="LoRA")

        cfg_path = os.path.join(lora_weights_path, "adapter_config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"LoRA folder is missing required file: {cfg_path}")
        weight_candidates = [
            os.path.join(lora_weights_path, "adapter_model.safetensors"),
            os.path.join(lora_weights_path, "adapter_model.bin"),
            os.path.join(lora_weights_path, "pytorch_model.bin"),
        ]
        if not any(os.path.isfile(p) for p in weight_candidates):
            raise FileNotFoundError(
                "LoRA folder is missing adapter weights. Expected one of: "
                + ", ".join(weight_candidates)
            )

        logger.info(f"Loading LoRA weights from: {lora_weights_path}")
        model.model = PeftModel.from_pretrained(
            model.model, 
            lora_weights_path,
            is_trainable=True 
        )
    
    # 2. 加载多模态组合权重 (Projector + Bio Updater)
    if mm_projector_path:
        _ensure_exists(mm_projector_path, kind="Multi-modal checkpoint")
        _ensure_is_file(mm_projector_path, kind="Multi-modal checkpoint")

        logger.info(f"Loading unified multi-modal weights from: {mm_projector_path}")
        device = next(model.parameters()).device
        checkpoint = torch.load(mm_projector_path, map_location=device)

        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Expected checkpoint dict in {mm_projector_path}, got: {type(checkpoint)}"
            )
        
        # 直接按组合格式加载
        _require_ckpt_key(checkpoint, "projector", path=mm_projector_path)
        model.projector.load_state_dict(checkpoint["projector"])
        logger.info("Loaded projector weights.")

        if hasattr(model, "bio_updater"):
            _require_ckpt_key(checkpoint, "bio_updater", path=mm_projector_path)
            model.bio_updater.load_state_dict(checkpoint["bio_updater"])
            logger.info("Loaded bio_updater weights.")
            if getattr(model, "bio_updater_gate", None) is not None and "bio_updater_gate" in checkpoint:
                model.bio_updater_gate.load_state_dict(checkpoint["bio_updater_gate"])
                logger.info("Loaded bio_updater_gate weights.")

        if hasattr(model, "bio_thinker"):
            _require_ckpt_key(checkpoint, "bio_thinker", path=mm_projector_path)
            model.bio_thinker.load_state_dict(checkpoint["bio_thinker"])
            logger.info("Loaded bio_thinker weights.")
            if getattr(model, "bio_thinker_gate", None) is not None and "bio_thinker_gate" in checkpoint:
                model.bio_thinker_gate.load_state_dict(checkpoint["bio_thinker_gate"])
                logger.info("Loaded bio_thinker_gate weights.")
        if hasattr(model, "task_thinker"):
            _require_ckpt_key(checkpoint, "task_thinker", path=mm_projector_path)
            model.task_thinker.load_state_dict(checkpoint["task_thinker"])
            logger.info("Loaded task_thinker weights.")
            if getattr(model, "task_thinker_gate", None) is not None and "task_thinker_gate" in checkpoint:
                model.task_thinker_gate.load_state_dict(checkpoint["task_thinker_gate"])
                logger.info("Loaded task_thinker_gate weights.")
            
    return model

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_stage3():
    parser = argparse.ArgumentParser(description="Stage 3 Training for Bio-LatentCOT")
    parser.add_argument("--data_path", type=str, default="/mnt/afs/L202500070/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot")
    parser.add_argument(
        "--qwen_size",
        type=str,
        default=ModelConfig.DEFAULT_QWEN_SIZE,
        help="Qwen backbone size. Allowed: 0.6b, 1.7b, 4b, 8b, 14b. Default: 8b.",
    )
    parser.add_argument("--lora_path", type=str, default=None, help="Stage 2 LoRA weights (optional)")
    parser.add_argument("--projector_path", type=str, default=None, help="Unified projector + bio_updater weights (optional)")
    parser.add_argument("--output_dir", type=str, default="./outputs/stage3_coconut")
    parser.add_argument("--epochs_per_stage", type=float, default=3, help="Number of epochs to train (per latent stage or for SFT)")
    parser.add_argument("--max_latent_stage", type=int, default=3, help="Max number of CoT steps to latent-ize")
    parser.add_argument("--c_thought", type=int, default=2, help="Number of latent tokens per CoT step")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Optional training seed. This only changes stage1-3 behavior when explicitly provided. "
            "If omitted, the current default behavior is preserved."
        ),
    )
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--save_full_model", type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to save full model weights (default False to save space)")
    parser.add_argument("--training_stage", type=int, default=3, choices=[1, 2, 3], help="Which stage to train: 1 (No COT), 2 (With COT), 3 (Latent/Coconut)")
    parser.add_argument(
        "--freeze_llm",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Freeze the text LLM weights (including any loaded LoRA adapters).",
    )
    parser.add_argument(
        "--freeze_projector",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Freeze the multi-modal projector module.",
    )
    parser.add_argument(
        "--freeze_bio_updater",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Freeze the bio_updater module (memory update).",
    )
    parser.add_argument(
        "--freeze_bioupdater_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Freeze the bio_updater gating module (Linear+Sigmoid hard switch).",
    )
    parser.add_argument(
        "--freeze_bio_thinker",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Freeze the bio_thinker module.",
    )
    parser.add_argument(
        "--freeze_biothinker_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Freeze the BioThinker gating module (Linear+Sigmoid hard switch).",
    )
    parser.add_argument(
        "--freeze_task_thinker",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Freeze the task_thinker module.",
    )
    parser.add_argument(
        "--freeze_taskthinker_gate",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Freeze the TaskThinker gating module (Linear+Sigmoid hard switch).",
    )
    # Stage 3 switches (only effective when --training_stage 3)
    parser.add_argument(
        "--is_coconut",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Whether to run Coconut latent training for stage 3 (ignored for stage 1/2).",
    )
    parser.add_argument(
        "--is_both_latent",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable Bio-latent thinker tokens for stage 3 (ignored for stage 1/2).",
    )
    parser.add_argument(
        "--is_biothinker",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable BioThinker (bio-latent block) when --is_both_latent is false.",
    )
    parser.add_argument(
        "--is_biothinker_multi",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Use BioThinkerMulti (4-expert weighted FFN) instead of BioThinker.",
    )
    parser.add_argument(
        "--is_taskthinker",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable TaskThinker (task-latent block) when --is_both_latent is false.",
    )
    parser.add_argument(
        "--taskthinker_type",
        type=str,
        default="mlp",
        choices=["mlp", "identity"],
        help="TaskThinker refinement type: 'mlp' (default) or 'identity' (no MLP / no gate, but keep the loop).",
    )
    parser.add_argument(
        "--is_taskthinker_multi",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Use TaskThinkerMulti (4-expert weighted MLP) instead of TaskThinker.",
    )
    parser.add_argument(
        "--is_bioupdater",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable BioUpdater (memory update) when --is_both_latent is false.",
    )
    parser.add_argument(
        "--is_bioupdater_multi",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Use BioTokenUpdaterMulti (4-expert weighted FFN) instead of BioTokenUpdater.",
    )
    parser.add_argument(
        "--is_bioupdater_gating",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable BioUpdater gating (Linear+Sigmoid hard switch). When false, behavior is unchanged.",
    )
    parser.add_argument(
        "--is_biothinker_gating",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable BioThinker gating (hard switch). When gate=0, bio-latent block is replaced by anchor embedding.",
    )
    parser.add_argument(
        "--is_taskthinker_gating",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Enable TaskThinker gating (hard switch). Gate scales the MLP residual: x + gate*y.",
    )
    parser.add_argument(
        "--bio_latent_lambda",
        type=float,
        default=0.0,
        help="Weight for bio-latent cosine hinge loss (only effective when --training_stage 3 and --is_both_latent true).",
    )
    parser.add_argument(
        "--bio_latent_alpha",
        type=float,
        default=0.5,
        help="Margin alpha for bio-latent cosine hinge loss: mean(max(0, alpha - cos(v, mu))).",
    )
    parser.add_argument(
        "--task_latent_lambda",
        type=float,
        default=0.0,
        help="Weight for task-latent prompt-alignment cosine hinge loss (only effective when --training_stage 3 and --is_both_latent true).",
    )
    parser.add_argument(
        "--task_latent_alpha",
        type=float,
        default=0.5,
        help="Margin alpha for task-latent prompt-alignment cosine hinge loss: mean(max(0, alpha - cos(v_prompt_last, mu_task))).",
    )
    parser.add_argument(
        "--bio_thinker_dropout",
        type=float,
        default=0.0,
        help="Dropout probability inside bio_thinker (TransformerEncoderLayer).",
    )
    parser.add_argument(
        "--task_thinker_dropout",
        type=float,
        default=0.0,
        help="Dropout probability inside task_thinker (MLP).",
    )
    parser.add_argument(
        "--max_cot_string_len",
        type=int,
        default=2048,
        help="Max CoT string length used to scale task-latent count: ceil(len(cot)/max_cot_string_len*4).",
    )
    parser.add_argument(
        "--task_latent_max_steps",
        type=int,
        default=10,
        help="Max loop steps when generating task latents during inference (get_prompt_embeddings).",
    )
    # Counterfactual bio-token embedding perturbation + loss
    parser.add_argument("--cf_lambda", type=float, default=0.0, help="Weight for counterfactual hinge loss (0 disables).")
    parser.add_argument("--cf_margin", type=float, default=0.5, help="Margin for hinge on (L_cf - L_pos).")
    parser.add_argument("--cf_prob", type=float, default=1.0, help="Probability of triggering a counterfactual paired-loss pass for a batch.")
    
    raw_argv = sys.argv[1:]
    args = parser.parse_args()
    seed_was_explicitly_provided = _arg_was_explicitly_provided(raw_argv, "--seed")
    qwen_model_path = _resolve_qwen_model_path(args.qwen_size)
    set_tokenizer_model_path(qwen_model_path)

    if seed_was_explicitly_provided:
        hf_set_seed(args.seed)

    # 1. 基础配置
    mol_config = {
        'num_queries': ModelConfig.NUM_QUERIES,
        'input_dim': ModelConfig.INPUT_DIM,
        'num_heads': ModelConfig.NUM_HEADS
    }
    
    # 当前的权重路径，初始为参数传入的路径
    current_lora_path = args.lora_path
    current_projector_path = args.projector_path

    # 2. 开启训练循环
    # Stage 1 & 2 只训练一次，Stage 3 开启分阶段循环训练
    if args.training_stage == 1:
        stages = [0]
        is_coconut = False
        is_both_latent = False
        is_biothinker = bool(args.is_biothinker)
        is_taskthinker = bool(args.is_taskthinker)
        taskthinker_type = str(args.taskthinker_type)
        is_bioupdater = bool(args.is_bioupdater)
        is_biothinker_multi = bool(args.is_biothinker_multi)
        is_taskthinker_multi = bool(args.is_taskthinker_multi)
        is_bioupdater_multi = bool(args.is_bioupdater_multi)
        is_bioupdater_gating = bool(args.is_bioupdater_gating)
        is_biothinker_gating = bool(args.is_biothinker_gating)
        is_taskthinker_gating = bool(args.is_taskthinker_gating)
        bio_latent_lambda = 0.0
        bio_latent_alpha = 0.5
        task_latent_lambda = 0.0
        task_latent_alpha = 0.5
        bio_thinker_dropout = 0.0
        task_thinker_dropout = 0.0
        max_cot_string_len = 2048
        task_latent_max_steps = 10
        include_cot = False
        mode_name = "Stage1-NoCOT"
    elif args.training_stage == 2:
        stages = [0]
        is_coconut = False
        is_both_latent = False
        is_biothinker = bool(args.is_biothinker)
        is_taskthinker = bool(args.is_taskthinker)
        taskthinker_type = str(args.taskthinker_type)
        is_bioupdater = bool(args.is_bioupdater)
        is_biothinker_multi = bool(args.is_biothinker_multi)
        is_taskthinker_multi = bool(args.is_taskthinker_multi)
        is_bioupdater_multi = bool(args.is_bioupdater_multi)
        is_bioupdater_gating = bool(args.is_bioupdater_gating)
        is_biothinker_gating = bool(args.is_biothinker_gating)
        is_taskthinker_gating = bool(args.is_taskthinker_gating)
        bio_latent_lambda = 0.0
        bio_latent_alpha = 0.5
        task_latent_lambda = 0.0
        task_latent_alpha = 0.5
        bio_thinker_dropout = 0.0
        task_thinker_dropout = 0.0
        max_cot_string_len = 2048
        task_latent_max_steps = 10
        include_cot = True
        mode_name = "Stage2-WithCOT"
    else: # Stage 3
        is_coconut = bool(args.is_coconut)
        is_both_latent = bool(args.is_both_latent)
        is_biothinker = bool(args.is_biothinker)
        is_taskthinker = bool(args.is_taskthinker)
        taskthinker_type = str(args.taskthinker_type)
        is_bioupdater = bool(args.is_bioupdater)
        is_biothinker_multi = bool(args.is_biothinker_multi)
        is_taskthinker_multi = bool(args.is_taskthinker_multi)
        is_bioupdater_multi = bool(args.is_bioupdater_multi)
        is_bioupdater_gating = bool(args.is_bioupdater_gating)
        is_biothinker_gating = bool(args.is_biothinker_gating)
        is_taskthinker_gating = bool(args.is_taskthinker_gating)
        bio_latent_lambda = float(args.bio_latent_lambda)
        bio_latent_alpha = float(args.bio_latent_alpha)
        task_latent_lambda = float(args.task_latent_lambda)
        task_latent_alpha = float(args.task_latent_alpha)
        bio_thinker_dropout = float(args.bio_thinker_dropout)
        task_thinker_dropout = float(args.task_thinker_dropout)
        max_cot_string_len = int(args.max_cot_string_len)
        task_latent_max_steps = int(args.task_latent_max_steps)
        include_cot = True
        if is_coconut:
            stages = range(args.max_latent_stage + 1)
            mode_name = "Stage3-Coconut"
        else:
            stages = [0]
            mode_name = "Stage3-WithCOT"

    for stage in stages:
        logger.info(f"\n" + "🚀" * 30)
        logger.info(f"STARTING {mode_name} (STAGE {stage})")
        if is_coconut:
            logger.info(f"Replace first {stage} steps with {stage * args.c_thought} latents")
        if is_both_latent:
            logger.info("Bio-latent thinker enabled (N_bio_latents = #smiles).")
        logger.info("🚀" * 30 + "\n")

        # 2.1 每一个 Stage 彻底重新初始化模型
        model = Qwen3MoleculeLLM(
            qwen_model_name=qwen_model_path,
            mol_config=mol_config,
            is_coconut=is_coconut,
            is_both_latent=is_both_latent,
            is_biothinker=is_biothinker,
            is_taskthinker=is_taskthinker,
            is_bioupdater=is_bioupdater,
            taskthinker_type=taskthinker_type,
            is_biothinker_multi=is_biothinker_multi,
            is_taskthinker_multi=is_taskthinker_multi,
            is_bioupdater_multi=is_bioupdater_multi,
            is_bioupdater_gating=is_bioupdater_gating,
            is_biothinker_gating=is_biothinker_gating,
            is_taskthinker_gating=is_taskthinker_gating,
            bio_latent_lambda=bio_latent_lambda,
            bio_latent_alpha=bio_latent_alpha,
            task_latent_lambda=task_latent_lambda,
            task_latent_alpha=task_latent_alpha,
            bio_thinker_dropout=bio_thinker_dropout,
            task_thinker_dropout=task_thinker_dropout,
            max_cot_string_len=max_cot_string_len,
            task_latent_max_steps=task_latent_max_steps,
        )
        tokenizer = model.tokenizer
        
        # 加载上一个 Stage 的权重
        if current_lora_path or current_projector_path:
            logger.info(f"Loading weights for training...")
            model = load_trained_components_stage3(
                model, 
                lora_weights_path=current_lora_path, 
                mm_projector_path=current_projector_path
            )
        
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

        if bool(args.freeze_llm):
            for param in model.model.parameters():
                param.requires_grad = False
            logger.info("freeze_llm=True: froze all LLM (base + LoRA) parameters.")
        
        # 确保投影器和 Bio Updater 可训练
        for param in model.projector.parameters():
            param.requires_grad = not bool(args.freeze_projector)
        
        bioupdater_enabled = bool(is_both_latent) or bool(is_bioupdater)
        for param in model.bio_updater.parameters():
            param.requires_grad = bioupdater_enabled and (not bool(args.freeze_bio_updater))
        if getattr(model, "bio_updater_gate", None) is not None:
            for param in model.bio_updater_gate.parameters():
                param.requires_grad = bioupdater_enabled and (not bool(args.freeze_bioupdater_gate))

        if hasattr(model, "bio_thinker"):
            biothinker_enabled = bool(is_both_latent) or bool(is_biothinker)
            for param in model.bio_thinker.parameters():
                param.requires_grad = biothinker_enabled and (not bool(args.freeze_bio_thinker))
        if getattr(model, "bio_thinker_gate", None) is not None:
            biothinker_enabled = bool(is_both_latent) or bool(is_biothinker)
            for param in model.bio_thinker_gate.parameters():
                param.requires_grad = biothinker_enabled and (not bool(args.freeze_biothinker_gate))
        if hasattr(model, "task_thinker"):
            taskthinker_enabled = bool(is_both_latent) or bool(is_taskthinker)
            for param in model.task_thinker.parameters():
                param.requires_grad = taskthinker_enabled and (not bool(args.freeze_task_thinker))
        if getattr(model, "task_thinker_gate", None) is not None:
            taskthinker_enabled = bool(is_both_latent) or bool(is_taskthinker)
            for param in model.task_thinker_gate.parameters():
                param.requires_grad = taskthinker_enabled and (not bool(args.freeze_taskthinker_gate))
        
        model.model.train()

        # 2.2 重新加载当前 Stage 的数据集
        train_dataset = load_data(
            args.data_path,
            include_cot=include_cot,
            is_coconut=is_coconut,
            scheduled_stage=stage,
            c_thought=args.c_thought,
            max_len=args.max_seq_length
        )

        # 2.3 配置当前 Stage 的输出目录
        stage_suffix = f"stage{args.training_stage}_sub{stage}" if is_coconut else f"stage{args.training_stage}"
        stage_output_dir = os.path.join(args.output_dir, stage_suffix)
        
        # 兼容不同版本的 TRL (0.15 vs 0.24+)
        sft_config_kwargs = {
            "output_dir": stage_output_dir,
            "num_train_epochs": args.epochs_per_stage,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "learning_rate": args.lr,
            "bf16": True,
            "remove_unused_columns": False,
            "logging_steps": 10,
            "save_strategy": "no",
            "save_total_limit": 1,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "ddp_find_unused_parameters": True,
            "report_to": "wandb",
            "optim": "adamw_8bit",
            "lr_scheduler_type": "cosine",
            "weight_decay": 0.01,
        }
        if seed_was_explicitly_provided:
            sft_config_kwargs["seed"] = args.seed
        
        # 检查 SFTConfig 支持哪个参数名 (max_seq_length 还是 max_length)
        if "max_seq_length" in inspect.signature(SFTConfig.__init__).parameters:
            sft_config_kwargs["max_seq_length"] = args.max_seq_length
        else:
            sft_config_kwargs["max_length"] = args.max_seq_length

        training_args = SFTConfig(**sft_config_kwargs)

        # WandB 记录
        if wandb.run is not None:
            wandb.finish() # 结束上一个 stage 的 run
        
        wandb_run_name = f"{mode_name}-sub{stage}" if is_coconut else mode_name
        wandb.init(
            project="qwen3-molecule-unified",
            name=f"{wandb_run_name}-{datetime.now().strftime('%m%d-%H%M')}",
            mode="offline",
            config={**vars(args), "current_stage": stage, "mode": mode_name}
        )

        data_collator = MultiModalDataCollator(tokenizer=tokenizer, model=model.model, padding=True)

        trainer = MultiModalSFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            cf_lambda=args.cf_lambda,
            cf_margin=args.cf_margin,
            cf_prob=args.cf_prob,
            callbacks=[LoraTrainingMonitorCallback(), TerminalPlotCallback()],
        )

        # 执行当前阶段训练
        trainer.train()
        
        # 2.4 保存当前阶段结果，并更新下个阶段的加载路径
        if args.save_full_model:
            logger.info("Saving full model weights for stage %s...", stage_suffix)
            trainer.save_model(stage_output_dir)
        else:
            logger.info("Skipping full model weights saving for stage %s (only saving LoRA and Projector).", stage_suffix)
            
        current_lora_path = os.path.join(stage_output_dir, "lora_weights")
        current_projector_path = os.path.join(stage_output_dir, "mm_projector.pt")
        
        # 手动保存 LoRA 和 组合后的多模态权重
        os.makedirs(current_lora_path, exist_ok=True)
        model.model.save_pretrained(current_lora_path)
        
        # 将 Projector / Bio Updater / Bio Thinker 存入同一个文件
        mm_weights = {
            'projector': model.projector.state_dict(),
            'bio_updater': model.bio_updater.state_dict(),
            'bio_thinker': model.bio_thinker.state_dict(),
            'task_thinker': model.task_thinker.state_dict(),
        }
        if getattr(model, "bio_updater_gate", None) is not None:
            mm_weights["bio_updater_gate"] = model.bio_updater_gate.state_dict()
        if getattr(model, "bio_thinker_gate", None) is not None:
            mm_weights["bio_thinker_gate"] = model.bio_thinker_gate.state_dict()
        if getattr(model, "task_thinker_gate", None) is not None:
            mm_weights["task_thinker_gate"] = model.task_thinker_gate.state_dict()
        torch.save(mm_weights, current_projector_path)
        
        # 如果不保存全模型，我们也至少存一下分词器，方便后续推理加载
        tokenizer.save_pretrained(stage_output_dir)
        
        logger.info(f"✅ {mode_name} Stage {stage} completed. Weights saved to {stage_output_dir}")
        
        # 显存清理
        del trainer, model, train_dataset
        torch.cuda.empty_cache()

    logger.info(f"🎉 All {mode_name} Stages completed!")

if __name__ == "__main__":
    train_stage3()
