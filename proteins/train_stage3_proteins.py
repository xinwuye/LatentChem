# train_stage3.py
import os
import sys
import math
import random
import torch
import torch.nn as nn
import argparse
import logging
import inspect
from datetime import datetime

from transformers import AutoTokenizer
import wandb
import plotext as plt

from model_stage3 import Qwen3MoleculeLLM
from proteins_dataloader import load_data, COCONUT_TOKENS  # keep your dataloader unchanged
from config_proteins import ModelConfig

from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            
            # 记录LoRA适配器信息（如果存在）
            if hasattr(model, 'peft_config') and model.peft_config is not None:
                try:
                    for adapter_name, config in model.peft_config.items():
                        logger.info(f"  LoRA配置 - {adapter_name}:")
                        logger.info(f"    r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
                except Exception:
                    # Some PEFT versions store differently; ignore if unable to introspect
                    pass
    
    def on_save(self, args, state, control, **kwargs):
        if wandb.run is not None:
            wandb.log({"checkpoint_step": state.global_step})
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if wandb.run is not None:
            wandb.log({"epoch": state.epoch})


class TerminalPlotCallback(TrainerCallback):
    def __init__(self):
        self.steps = []
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            if state.is_world_process_zero:
                self.steps.append(state.global_step)
                self.losses.append(logs["loss"])
                
                display_steps = self.steps[-100:]
                display_losses = self.losses[-100:]

                plt.clf()
                plt.plot(display_steps, display_losses, marker="dot", color="red", label="SFT Loss")
                plt.title("Real-time Training Loss (Terminal)")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.plotsize(100, 25)
                plt.grid(True)
                plt.show()


# class MultiModalDataCollator(DataCollatorForSeq2Seq):
#     def __init__(self, tokenizer, model, padding=True):
#         super().__init__(tokenizer=tokenizer, model=model, padding=padding)
#     def __call__(self, features):
#         smiles = [f.pop("smiles") for f in features]
#         cot_len = [f.pop("cot_len") for f in features] if features and ("cot_len" in features[0]) else None
#         batch = super().__call__(features)
#         batch["smiles"] = smiles
#         if cot_len is not None:
#             batch["cot_len"] = torch.tensor(cot_len, dtype=torch.long)
#         return batch

# Industrial-grade multimodal data collator (robust to proteins vs SMILES)
class MultiModalDataCollator(DataCollatorForSeq2Seq):
    """
    Robust multimodal collator:
     - accepts examples from molecule workflows ('smiles') or protein loader ('input_proteins')
     - strips non-tokenizer fields (like 'metadata') before calling tokenizer.pad
     - re-attaches 'smiles' and optional 'cot_len' afterwards
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _extract_smiles_like(self, features):
        # Prefer explicit 'smiles'
        if "smiles" in features[0]:
            smiles = [f.pop("smiles") for f in features]
            return smiles

        # Fallback to protein loader 'input_proteins'
        if "input_proteins" in features[0]:
            raw = [f.pop("input_proteins") for f in features]
            normalized = []
            for entry in raw:
                if entry is None:
                    normalized.append([])
                elif isinstance(entry, str):
                    normalized.append([entry])
                elif isinstance(entry, (list, tuple)):
                    normalized.append([s for s in entry if isinstance(s, str)])
                else:
                    normalized.append([])
            return normalized

        # Additional fallbacks
        if "protein" in features[0] or "proteins" in features[0]:
            key = "protein" if "protein" in features[0] else "proteins"
            raw = [f.pop(key) for f in features]
            normalized = []
            for entry in raw:
                if entry is None:
                    normalized.append([])
                elif isinstance(entry, str):
                    normalized.append([entry])
                elif isinstance(entry, (list, tuple)):
                    normalized.append([s for s in entry if isinstance(s, str)])
                else:
                    normalized.append([])
            return normalized

        raise KeyError(
            "Batch examples do not contain 'smiles' or 'input_proteins'. "
            f"Available keys: {list(features[0].keys())}"
        )

    def __call__(self, features):
        # 1) Normalize smiles-like lists and pop them out so tokenizer won't see them
        smiles = self._extract_smiles_like(features)

        # 2) Extract cot_len if present and remove from features
        cot_len = None
        if features and ("cot_len" in features[0]):
            cot_len = [f.pop("cot_len") for f in features]

        # 3) Build a trimmed features list that contains only keys the tokenizer expects
        # Typical token keys: input_ids, attention_mask, labels, decoder_input_ids (if any)
        allowed_keys = {"input_ids", "attention_mask", "labels", "decoder_input_ids"}
        features_for_tokenizer = []
        for f in features:
            ff = {k: v for k, v in f.items() if k in allowed_keys}
            # sanity: ensure input_ids exists (otherwise tokenizer.pad might fail later)
            if "input_ids" not in ff and hasattr(self, "tokenizer") and hasattr(self.tokenizer, "encode"):
                # if features previously had 'text' or 'query' fields you may need to tokenize earlier;
                # just raise to avoid silent errors
                raise KeyError("Feature missing 'input_ids'. Make sure your dataset returns tokenized fields.")
            features_for_tokenizer.append(ff)

        # 4) Call parent collator on the trimmed features
        batch = super().__call__(features_for_tokenizer)

        # 5) Re-attach smiles and cot_len into the returned batch
        batch["smiles"] = smiles
        if cot_len is not None:
            batch["cot_len"] = torch.tensor(cot_len, dtype=torch.long)

        return batch


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
        def _out_get(outputs, key, default=None):
            if isinstance(outputs, dict):
                return outputs.get(key, default)
            return getattr(outputs, key, default)

        def _is_world_zero():
            fn = getattr(self, "is_world_process_zero", None)
            return fn() if callable(fn) else bool(fn)

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

    # 1. LoRA
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
    
    # 2. projector + components
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
        
        _require_ckpt_key(checkpoint, "projector", path=mm_projector_path)
        model.projector.load_state_dict(checkpoint["projector"])
        logger.info("Loaded projector weights.")

        if hasattr(model, "bio_updater"):
            _require_ckpt_key(checkpoint, "bio_updater", path=mm_projector_path)
            model.bio_updater.load_state_dict(checkpoint["bio_updater"])
            logger.info("Loaded bio_updater weights.")

        if hasattr(model, "bio_thinker"):
            _require_ckpt_key(checkpoint, "bio_thinker", path=mm_projector_path)
            model.bio_thinker.load_state_dict(checkpoint["bio_thinker"])
            logger.info("Loaded bio_thinker weights.")
        if hasattr(model, "task_thinker"):
            _require_ckpt_key(checkpoint, "task_thinker", path=mm_projector_path)
            model.task_thinker.load_state_dict(checkpoint["task_thinker"])
            logger.info("Loaded task_thinker weights.")
            
    return model


def train_stage3():
    parser = argparse.ArgumentParser(description="Stage 3 Training for Bio-LatentCOT")
    parser.add_argument("--data_path", type=str, default="Mol-Instructions/data/proteins/Protein-oriented_Instructions")
    parser.add_argument("--encoder_type", type=str, default=getattr(ModelConfig, "ENCODER_TYPE", "protein"), choices=["smiles", "protein"])
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--projector_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/stage3_coconut")
    parser.add_argument("--epochs_per_stage", type=float, default=3)
    parser.add_argument("--max_latent_stage", type=int, default=3)
    parser.add_argument("--c_thought", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--save_full_model", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--training_stage", type=int, default=3, choices=[1,2,3])
    parser.add_argument("--freeze_llm", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--freeze_projector", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--freeze_bio_updater", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--freeze_bio_thinker", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--freeze_task_thinker", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--is_coconut", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--is_both_latent", type=lambda x: (str(x).lower() == "true"), default=False)
    parser.add_argument("--bio_latent_lambda", type=float, default=0.0)
    parser.add_argument("--bio_latent_alpha", type=float, default=0.5)
    parser.add_argument("--task_latent_lambda", type=float, default=0.0)
    parser.add_argument("--task_latent_alpha", type=float, default=0.5)
    parser.add_argument("--bio_thinker_dropout", type=float, default=0.0)
    parser.add_argument("--task_thinker_dropout", type=float, default=0.0)
    parser.add_argument("--max_cot_string_len", type=int, default=2048)
    parser.add_argument("--task_latent_max_steps", type=int, default=10)
    parser.add_argument("--cf_lambda", type=float, default=0.0)
    parser.add_argument("--cf_margin", type=float, default=0.5)
    parser.add_argument("--cf_prob", type=float, default=1.0)

    args = parser.parse_args()

    mol_config = {
        'num_queries': ModelConfig.NUM_QUERIES,
        'input_dim': ModelConfig.INPUT_DIM,
        'num_heads': ModelConfig.NUM_HEADS,
        'encoder_type': args.encoder_type,
        'smi_ted_folder': getattr(ModelConfig, "DEFAULT_SMI_TED_FOLDER", "./smi_ted_light"),
        'smi_ted_ckpt': getattr(ModelConfig, "DEFAULT_SMI_TED_CKPT", "smi-ted-Light_40.pt"),
        'smi_ted_vocab': getattr(ModelConfig, "DEFAULT_SMI_TED_VOCAB", "bert_vocab_curated.txt"),
        'max_len': getattr(ModelConfig, "MAX_PROTEIN_LEN", 2048),
    }

    # Determine training schedule per stage
    if args.training_stage == 1:
        stages = [0]
        include_cot = False
        is_coconut = False
        is_both_latent = False
        mode_name = "Stage1-NoCOT"
    elif args.training_stage == 2:
        stages = [0]
        include_cot = True
        is_coconut = False
        is_both_latent = False
        mode_name = "Stage2-WithCOT"
    else:
        is_coconut = bool(args.is_coconut)
        is_both_latent = bool(args.is_both_latent)
        include_cot = True
        if is_coconut:
            stages = range(args.max_latent_stage + 1)
            mode_name = "Stage3-Coconut"
        else:
            stages = [0]
            mode_name = "Stage3-WithCOT"

    current_lora_path = args.lora_path
    current_projector_path = args.projector_path

    for stage in stages:
        logger.info(f"\n{'='*40}\nSTARTING {mode_name} (STAGE {stage})\n{'='*40}")
        if is_coconut:
            logger.info(f"Replace first {stage} steps with {stage * args.c_thought} latents")
        if is_both_latent:
            logger.info("Bio-latent thinker enabled (N_bio_latents = #smiles).")

        # instantiate model
        model = Qwen3MoleculeLLM(
            qwen_model_name=ModelConfig.DEFAULT_QWEN_PATH,
            mol_config=mol_config,
            is_coconut=is_coconut,
            is_both_latent=is_both_latent,
            bio_latent_lambda=args.bio_latent_lambda,
            bio_latent_alpha=args.bio_latent_alpha,
            task_latent_lambda=args.task_latent_lambda,
            task_latent_alpha=args.task_latent_alpha,
            bio_thinker_dropout=args.bio_thinker_dropout,
            task_thinker_dropout=args.task_thinker_dropout,
            max_cot_string_len=args.max_cot_string_len,
            task_latent_max_steps=args.task_latent_max_steps,
        )
        tokenizer = model.tokenizer

        # load previous weights if provided
        if current_lora_path or current_projector_path:
            logger.info("Loading provided weights...")
            model = load_trained_components_stage3(
                model,
                lora_weights_path=current_lora_path,
                mm_projector_path=current_projector_path,
            )

        # configure LoRA if not present
        if not hasattr(model.model, 'peft_config') or model.model.peft_config is None:
            logger.info("Configuring LoRA from scratch...")
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

        # projector / bio updater trainability
        for param in model.projector.parameters():
            param.requires_grad = not bool(args.freeze_projector)
        for param in model.bio_updater.parameters():
            param.requires_grad = not bool(args.freeze_bio_updater)
        if hasattr(model, "bio_thinker"):
            for param in model.bio_thinker.parameters():
                param.requires_grad = bool(is_both_latent) and (not bool(args.freeze_bio_thinker))
        if hasattr(model, "task_thinker"):
            for param in model.task_thinker.parameters():
                param.requires_grad = bool(is_both_latent) and (not bool(args.freeze_task_thinker))

        model.model.train()

        # load data using your protein-only dataloader (unchanged)
        train_dataset = load_data(
            args.data_path,
            include_cot=include_cot,
            is_coconut=is_coconut,
            scheduled_stage=stage,
            c_thought=args.c_thought,
        )

        stage_suffix = f"stage{args.training_stage}_sub{stage}" if is_coconut else f"stage{args.training_stage}"
        stage_output_dir = os.path.join(args.output_dir, stage_suffix)

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

        if "max_seq_length" in inspect.signature(SFTConfig.__init__).parameters:
            sft_config_kwargs["max_seq_length"] = args.max_seq_length
        else:
            sft_config_kwargs["max_length"] = args.max_seq_length

        training_args = SFTConfig(**sft_config_kwargs)

        # finish wandb runs
        if wandb.run is not None:
            wandb.finish()

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
            train_dataset=train_dataset['train'],
            processing_class=tokenizer,
            data_collator=data_collator,
            cf_lambda=args.cf_lambda,
            cf_margin=args.cf_margin,
            cf_prob=args.cf_prob,
            callbacks=[LoraTrainingMonitorCallback(), TerminalPlotCallback()],
        )

        trainer.train()

        if args.save_full_model:
            logger.info("Saving full model weights for stage %s...", stage_suffix)
            trainer.save_model(stage_output_dir)
        else:
            logger.info("Skipping full model weights saving for stage %s (only saving LoRA and Projector).", stage_suffix)

        current_lora_path = os.path.join(stage_output_dir, "lora_weights")
        current_projector_path = os.path.join(stage_output_dir, "mm_projector.pt")
        os.makedirs(current_lora_path, exist_ok=True)
        model.model.save_pretrained(current_lora_path)

        mm_weights = {
            'projector': model.projector.state_dict(),
            'bio_updater': model.bio_updater.state_dict(),
            'bio_thinker': model.bio_thinker.state_dict(),
            'task_thinker': model.task_thinker.state_dict(),
        }
        torch.save(mm_weights, current_projector_path)
        tokenizer.save_pretrained(stage_output_dir)

        logger.info(f"✅ {mode_name} Stage {stage} completed. Weights saved to {stage_output_dir}")

        del trainer, model, train_dataset
        torch.cuda.empty_cache()

    logger.info(f"🎉 All {mode_name} Stages completed!")


if __name__ == "__main__":
    train_stage3()