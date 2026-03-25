import os
import argparse
import logging
from datetime import datetime

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import TrainerCallback

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore[assignment]

try:
    import plotext as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

from config import ModelConfig
from dataloader import load_grpo_data, set_tokenizer_model_path
from model_stage3 import Qwen3MoleculeLLM
from trainer_try2.grpo_trainer import QwenMoleculeGRPOTrainer
from trainer_try2.grpo_config import GRPOConfig
from trainer_try2.reward_func import (
    format_reward_answer_tag,
    reward_answer_correctness,
    reward_answer_correctness_bench,
    reward_answer_type_validity,
    reward_stage4_corrupt_or_correct,
    reward_stage4_double_scaled_correctness,
    reward_stage4_scaled_correctness,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_qwen_model_path(qwen_size: str) -> str:
    return ModelConfig.require_qwen_path(qwen_size)

def _find_latest_checkpoint(run_dir: str) -> str | None:
    if not run_dir or (not os.path.isdir(run_dir)):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(run_dir):
        if not name.startswith("checkpoint-"):
            continue
        step_str = name[len("checkpoint-") :]
        if not step_str.isdigit():
            continue
        path = os.path.join(run_dir, name)
        if not os.path.isdir(path):
            continue
        step = int(step_str)
        if best_step is None or step > best_step:
            best_step = step
            best_path = path
    return best_path


class TerminalRewardPlotCallback(TrainerCallback):
    def __init__(self, reward_key: str = "reward"):
        self.reward_key = str(reward_key)
        self.steps: list[int] = []
        self.rewards: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if plt is None:
            return
        if not logs or self.reward_key not in logs:
            return
        if not state.is_world_process_zero:
            return

        try:
            step = int(state.global_step)
            val = float(logs[self.reward_key])
        except Exception:
            return

        self.steps.append(step)
        self.rewards.append(val)

        display_steps = self.steps[-100:]
        display_rewards = self.rewards[-100:]

        plt.clf()
        plt.plot(display_steps, display_rewards, marker="dot", color="green", label=self.reward_key)
        plt.title("Real-time GRPO Reward (Terminal)")
        plt.xlabel("Step")
        plt.ylabel(self.reward_key)
        plt.plotsize(100, 25)
        plt.grid(True)
        plt.show()


def _ensure_lora_and_trainables(
    model: Qwen3MoleculeLLM,
    *,
    freeze_llm: bool = False,
    freeze_projector: bool = False,
    freeze_bio_updater: bool = False,
    freeze_bioupdater_gate: bool = False,
    freeze_bio_thinker: bool = False,
    freeze_biothinker_gate: bool = False,
    freeze_task_thinker: bool = False,
    freeze_taskthinker_gate: bool = False,
):
    """
    Make sure LoRA is enabled on the underlying text model, and that the multimodal heads are trainable.
    Mirrors the training intent of `train_stage3.py`, but for GRPO.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Enable / create LoRA on the base LLM
    if not hasattr(model.model, "peft_config") or model.model.peft_config is None:
        logger.info("Configuring LoRA from scratch...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model.model = get_peft_model(model.model, lora_config)

    # LLM / LoRA trainability
    if not freeze_llm:
        # IMPORTANT: we froze everything above, which also freezes LoRA params loaded from checkpoint.
        # GRPO needs the policy parameters (LoRA adapters) to require grad; otherwise loss will be detached.
        lora_param_count = 0
        for name, p in model.model.named_parameters():
            if "lora_" in name or "modules_to_save" in name:
                p.requires_grad = True
                lora_param_count += p.numel()
        if lora_param_count == 0:
            logger.warning("No LoRA parameters were marked trainable; GRPO may fail (detached loss).")
    else:
        logger.info("freeze_llm=True: keeping all LLM / LoRA parameters frozen.")

    # Multimodal heads trainability
    if not freeze_projector:
        for p in model.projector.parameters():
            p.requires_grad = True
    both_latent = bool(model.is_both_latent)
    bioupdater_enabled = both_latent or bool(model.is_bioupdater)
    biothinker_enabled = both_latent or bool(model.is_biothinker)
    taskthinker_enabled = both_latent or bool(model.is_taskthinker)

    if bioupdater_enabled and (not freeze_bio_updater):
        for p in model.bio_updater.parameters():
            p.requires_grad = True
    if bioupdater_enabled and (not freeze_bioupdater_gate) and getattr(model, "bio_updater_gate", None) is not None:
        for p in model.bio_updater_gate.parameters():
            p.requires_grad = True

    if biothinker_enabled and hasattr(model, "bio_thinker") and (not freeze_bio_thinker):
        for p in model.bio_thinker.parameters():
            p.requires_grad = True
    if biothinker_enabled and (not freeze_biothinker_gate) and getattr(model, "bio_thinker_gate", None) is not None:
        for p in model.bio_thinker_gate.parameters():
            p.requires_grad = True
    if taskthinker_enabled and hasattr(model, "task_thinker") and (not freeze_task_thinker):
        for p in model.task_thinker.parameters():
            p.requires_grad = True
    if taskthinker_enabled and (not freeze_taskthinker_gate) and getattr(model, "task_thinker_gate", None) is not None:
        for p in model.task_thinker_gate.parameters():
            p.requires_grad = True


def load_trained_components_stage3(model, lora_weights_path=None, mm_projector_path=None):
    """
    Same checkpoint format as `train_stage3.py`:
    - LoRA weights folder
    - Combined projector + bio_updater file (mm_projector.pt)
    """
    if lora_weights_path and os.path.exists(lora_weights_path):
        logger.info(f"Loading LoRA weights from: {lora_weights_path}")
        model.model = PeftModel.from_pretrained(model.model, lora_weights_path, is_trainable=True)

    if mm_projector_path and os.path.exists(mm_projector_path):
        logger.info(f"Loading unified multi-modal weights from: {mm_projector_path}")
        device = next(model.parameters()).device
        checkpoint = torch.load(mm_projector_path, map_location=device)
        model.projector.load_state_dict(checkpoint["projector"])
        model.bio_updater.load_state_dict(checkpoint.get("bio_updater", {}), strict=False)
        if getattr(model, "bio_updater_gate", None) is not None:
            model.bio_updater_gate.load_state_dict(checkpoint.get("bio_updater_gate", {}), strict=False)
        if hasattr(model, "bio_thinker"):
            model.bio_thinker.load_state_dict(checkpoint.get("bio_thinker", {}), strict=False)
        if getattr(model, "bio_thinker_gate", None) is not None:
            model.bio_thinker_gate.load_state_dict(checkpoint.get("bio_thinker_gate", {}), strict=False)
        if hasattr(model, "task_thinker"):
            model.task_thinker.load_state_dict(checkpoint.get("task_thinker", {}), strict=False)
        if getattr(model, "task_thinker_gate", None) is not None:
            model.task_thinker_gate.load_state_dict(checkpoint.get("task_thinker_gate", {}), strict=False)
        logger.info("Loaded projector (+ bio_updater if present).")

    return model


def main():
    parser = argparse.ArgumentParser(description="GRPO try1 training for Bio-LatentCOT (smiles-aware, optional vLLM).")
    parser.add_argument(
        "--qwen_size",
        type=str,
        default=ModelConfig.DEFAULT_QWEN_SIZE,
        help="Qwen backbone size. Allowed: 0.6b, 1.7b, 4b, 8b, 14b. Default: 8b.",
    )
    parser.add_argument(
        "--use_reward_answer_tag",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `format_reward_answer_tag` in reward functions.",
    )
    parser.add_argument(
        "--use_reward_answer_type_validity",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `reward_answer_type_validity` in reward functions.",
    )
    parser.add_argument(
        "--use_reward_answer_correctness_bench",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `reward_answer_correctness_bench` in reward functions.",
    )
    parser.add_argument(
        "--use_reward_answer_correctness",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include legacy `reward_answer_correctness` in reward functions (non-benchmark routing).",
    )
    parser.add_argument(
        "--use_reward_stage4_corrupt_or_correct",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `reward_stage4_corrupt_or_correct` in reward functions (uses `corrupt_prob`).",
    )
    parser.add_argument(
        "--use_reward_stage4_scaled_correctness",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `reward_stage4_scaled_correctness` in reward functions (uses `cot_len`).",
    )
    parser.add_argument(
        "--use_reward_stage4_double_scaled_correctness",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include `reward_stage4_double_scaled_correctness` in reward functions (uses `task_latent_count` + `cot_len`).",
    )
    parser.add_argument("--data_path", type=str, default=ModelConfig.DEFAULT_DATA_PATH)

    # Freeze switches (mirrors train_stage3.py)
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

    # Load starting weights (optional)
    parser.add_argument("--lora_path", type=str, default=None, help="Initial LoRA weights folder (optional)")
    parser.add_argument("--projector_path", type=str, default=None, help="Initial mm_projector.pt (optional)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo_try1")
    parser.add_argument("--run_name", type=str, default=None)

    # Logging / Weights & Biases
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Trainer reporters (e.g. 'wandb' or 'none'). Default: wandb.",
    )
    parser.add_argument("--wandb_project", type=str, default="biolatent-grpo")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="offline",
        choices=["offline", "online", "disabled"],
        help="W&B mode. 'disabled' forces report_to='none'.",
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default=None,
        help="W&B log dir. Defaults to `<repo>/code_train_sft/wandb`.",
    )

    # Training
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from a checkpoint dir, or set to 'latest' to auto-pick the newest checkpoint under output_dir/run_name.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=256)

    # Stage 4: task-latent corruption
    parser.add_argument(
        "--corrupt_prob",
        type=float,
        default=0.0,
        help="Probability to corrupt task latent embeddings per prompt group (stage=4/5 only).",
    )
    parser.add_argument(
        "--corrupt_latent_noise_std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise to replace task latent embeddings (0 -> zeros) (stage=4/5 only).",
    )
    parser.add_argument(
        "--is_both_latent",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Enable task-latent generation via is_both_latent (stage=4/5 requires true).",
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

    # GRPO
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument(
        "--steps_per_generation",
        type=int,
        default=None,
        help="How many training steps to reuse one rollout batch. If omitted, GRPO defaults to `grad_accum`.",
    )
    parser.add_argument("--beta", type=float, default=0.0, help="KL beta (0 disables ref model).")
    parser.add_argument("--epsilon", type=float, default=0.2)

    # Sampling
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Efficiency
    parser.add_argument("--use_liger", action="store_true", help="Use Liger Kernel for memory efficient training.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")

    # vLLM
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"])
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--vllm_ckpt",
        type=str,
        default=None,
        help="Optional vLLM base model checkpoint path/name. Defaults to the resolved --qwen_size path.",
    )
    parser.add_argument("--vllm_max_model_len", type=int, default=4096, help="Maximum model length for vLLM engine.")

    args = parser.parse_args()
    qwen_model_path = _resolve_qwen_model_path(args.qwen_size)
    set_tokenizer_model_path(qwen_model_path)
    if args.vllm_ckpt is None:
        args.vllm_ckpt = qwen_model_path

    run_name = args.run_name or f"grpo_try1-{datetime.now().strftime('%m%d-%H%M')}"
    os.makedirs(args.output_dir, exist_ok=True)

    report_to = str(args.report_to or "none")
    if str(args.wandb_mode).lower() == "disabled":
        report_to = "none"
    if "wandb" in report_to.lower():
        if wandb is None:
            raise ImportError("W&B is enabled (report_to includes 'wandb') but `wandb` is not installed.")
        wandb_dir = args.wandb_dir
        if not wandb_dir:
            wandb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(
            project=str(args.wandb_project),
            entity=(str(args.wandb_entity) if args.wandb_entity else None),
            name=run_name,
            mode=str(args.wandb_mode).lower(),
            dir=wandb_dir,
            config=vars(args),
        )

    # 1) Build model
    mol_config = {
        "num_queries": ModelConfig.NUM_QUERIES,
        "input_dim": ModelConfig.INPUT_DIM,
        "num_heads": ModelConfig.NUM_HEADS,
    }
    use_both_latent = bool(args.is_both_latent)
    enable_corruption = float(args.corrupt_prob) > 0.0
    if enable_corruption and (not use_both_latent):
        raise ValueError("corrupt_prob>0 requires --is_both_latent true (task latents must exist to corrupt).")
    model = Qwen3MoleculeLLM(
        qwen_model_name=qwen_model_path,
        mol_config=mol_config,
        is_both_latent=use_both_latent,
        is_biothinker=bool(args.is_biothinker),
        is_taskthinker=bool(args.is_taskthinker),
        is_bioupdater=bool(args.is_bioupdater),
        taskthinker_type=str(args.taskthinker_type),
        is_biothinker_multi=bool(args.is_biothinker_multi),
        is_taskthinker_multi=bool(args.is_taskthinker_multi),
        is_bioupdater_multi=bool(args.is_bioupdater_multi),
        is_bioupdater_gating=bool(args.is_bioupdater_gating),
        is_biothinker_gating=bool(args.is_biothinker_gating),
        is_taskthinker_gating=bool(args.is_taskthinker_gating),
        is_coconut=False,
        bio_thinker_dropout=float(args.bio_thinker_dropout),
        task_thinker_dropout=float(args.task_thinker_dropout),
    )

    # 2) Load weights if provided
    if args.lora_path or args.projector_path:
        model = load_trained_components_stage3(model, args.lora_path, args.projector_path)

    # 3) Ensure trainables
    _ensure_lora_and_trainables(
        model,
        freeze_llm=bool(args.freeze_llm),
        freeze_projector=bool(args.freeze_projector),
        freeze_bio_updater=bool(args.freeze_bio_updater),
        freeze_bioupdater_gate=bool(args.freeze_bioupdater_gate),
        freeze_bio_thinker=bool(args.freeze_bio_thinker),
        freeze_biothinker_gate=bool(args.freeze_biothinker_gate),
        freeze_task_thinker=bool(args.freeze_task_thinker),
        freeze_taskthinker_gate=bool(args.freeze_taskthinker_gate),
    )

    # 4) Dataset
    train_dataset = load_grpo_data(args.data_path)

    # 5) GRPO config
    grpo_args = GRPOConfig(
        output_dir=os.path.join(args.output_dir, run_name),
        run_name=run_name,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        ddp_find_unused_parameters=True,
        # Some submodules (e.g. frozen molecule encoder) can contain inference-mode buffers; broadcasting them under
        # DDP may error with "Inplace update to inference tensor". LLM training does not rely on buffer sync.
        ddp_broadcast_buffers=False,
        remove_unused_columns=False,
        report_to=report_to,
        seed=args.seed,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        steps_per_generation=args.steps_per_generation,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon,
        loss_type="grpo",
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_ckpt=args.vllm_ckpt,
        vllm_max_model_length=args.vllm_max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        use_liger_kernel=False,
        use_liger_manual=args.use_liger,
    )

    reward_funcs = []
    if bool(args.use_reward_answer_tag):
        reward_funcs.append(format_reward_answer_tag)
    if bool(args.use_reward_answer_type_validity):
        reward_funcs.append(reward_answer_type_validity)
    if bool(args.use_reward_answer_correctness):
        reward_funcs.append(reward_answer_correctness)
    if bool(args.use_reward_answer_correctness_bench):
        reward_funcs.append(reward_answer_correctness_bench)
    if bool(args.use_reward_stage4_corrupt_or_correct):
        reward_funcs.append(reward_stage4_corrupt_or_correct)
    if bool(args.use_reward_stage4_scaled_correctness):
        reward_funcs.append(reward_stage4_scaled_correctness)
    if bool(args.use_reward_stage4_double_scaled_correctness):
        reward_funcs.append(reward_stage4_double_scaled_correctness)

    if not reward_funcs:
        raise ValueError("No reward functions selected. Set at least one `--use_reward_* true` flag.")

    corrupt_prob = float(args.corrupt_prob) if enable_corruption else 0.0
    corrupt_latent_noise_std = float(args.corrupt_latent_noise_std) if enable_corruption else 0.0
    training_stage = 4 if enable_corruption else 3

    trainer = QwenMoleculeGRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        processing_class=model.tokenizer,
        callbacks=[TerminalRewardPlotCallback()],
        training_stage=int(training_stage),
        corrupt_prob=corrupt_prob,
        corrupt_latent_noise_std=corrupt_latent_noise_std,
    )

    resume = args.resume_from_checkpoint
    if resume:
        resume_s = str(resume).strip()
        if resume_s.lower() == "latest":
            ckpt = _find_latest_checkpoint(grpo_args.output_dir)
            if ckpt is None:
                raise ValueError(f"--resume_from_checkpoint latest: no checkpoint-* directory found under {grpo_args.output_dir}")
            resume_s = ckpt
        if not os.path.isdir(resume_s):
            raise ValueError(f"--resume_from_checkpoint path does not exist or is not a directory: {resume_s}")
        logger.info("Resuming training from checkpoint: %s", resume_s)
        trainer.train(resume_from_checkpoint=resume_s)
    else:
        trainer.train()

    # Save final LoRA + multimodal heads (compatible with stage checkpoints)
    final_dir = grpo_args.output_dir
    lora_dir = os.path.join(final_dir, "lora_weights")
    os.makedirs(lora_dir, exist_ok=True)
    model.model.save_pretrained(lora_dir)
    mm_path = os.path.join(final_dir, "mm_projector.pt")
    to_save = {"projector": model.projector.state_dict(), "bio_updater": model.bio_updater.state_dict()}
    if getattr(model, "bio_updater_gate", None) is not None:
        to_save["bio_updater_gate"] = model.bio_updater_gate.state_dict()
    if hasattr(model, "bio_thinker"):
        to_save["bio_thinker"] = model.bio_thinker.state_dict()
    if getattr(model, "bio_thinker_gate", None) is not None:
        to_save["bio_thinker_gate"] = model.bio_thinker_gate.state_dict()
    if hasattr(model, "task_thinker"):
        to_save["task_thinker"] = model.task_thinker.state_dict()
    if getattr(model, "task_thinker_gate", None) is not None:
        to_save["task_thinker_gate"] = model.task_thinker_gate.state_dict()
    torch.save(to_save, mm_path)
    model.tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved LoRA to {lora_dir} and mm weights to {mm_path}")
    if wandb is not None and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
