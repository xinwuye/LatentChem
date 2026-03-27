import argparse
import logging
import os
import sys
from datetime import datetime

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import TrainerCallback, set_seed as hf_set_seed

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
from model_text import load_qwen3_text_model
from trainer_try2.grpo_config import GRPOConfig
from trainer_try2.reward_func import (
    format_reward_answer_tag,
    reward_answer_correctness,
    reward_answer_correctness_bench,
    reward_answer_correctness_bench_output_cot_scaled,
    reward_answer_tag_output_cot_scaled,
    reward_answer_type_validity,
    reward_answer_type_validity_output_cot_scaled,
)
from trainer_try2.text_grpo_trainer import TextGRPOTrainer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_qwen_model_path(qwen_size: str) -> str:
    return ModelConfig.require_qwen_path(qwen_size)


def _arg_was_explicitly_provided(argv: list[str], flag: str) -> bool:
    if not flag.startswith("--"):
        raise ValueError(f"Expected a long CLI flag like '--seed', got: {flag}")
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


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


def _ensure_exists(path: str, *, kind: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")


def _ensure_is_dir(path: str, *, kind: str) -> None:
    if not os.path.isdir(path):
        raise ValueError(f"{kind} must be a directory: {path}")


def _prepare_text_lora_model(model, *, lora_path: str | None, lora_r: int, lora_alpha: int, lora_dropout: float):
    if lora_path:
        _ensure_exists(lora_path, kind="LoRA weights")
        _ensure_is_dir(lora_path, kind="LoRA weights")
        logger.info("Loading LoRA weights from: %s", lora_path)
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
        return model

    if hasattr(model, "peft_config") and model.peft_config is not None:
        return model

    logger.info("Configuring LoRA from scratch...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser(description="GRPO training for plain text-only Qwen3.")
    parser.add_argument(
        "--qwen_size",
        type=str,
        default=ModelConfig.DEFAULT_QWEN_SIZE,
        help="Qwen backbone size. Allowed: 0.6b, 1.7b, 4b, 8b, 14b. Default: 8b.",
    )
    parser.add_argument("--data_path", type=str, default=ModelConfig.DEFAULT_DATA_PATH)
    parser.add_argument("--lora_path", type=str, default=None, help="Initial LoRA weights folder (optional)")
    parser.add_argument("--output_dir", type=str, default="./outputs/grpo_text")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument(
        "--log_reward_trace",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help=(
            "If true, write per-rollout reward traces as rank-sharded JSONL files. "
            "Each line records task/subtask and every subreward value."
        ),
    )
    parser.add_argument(
        "--reward_trace_dir",
        type=str,
        default=None,
        help=(
            "Optional directory for per-rank reward trace JSONL files. "
            "Defaults to <output_dir>/<run_name>/reward_trace when --log_reward_trace true."
        ),
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--steps_per_generation", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_liger", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["server", "colocate"])
    parser.add_argument("--vllm_ckpt", type=str, default=None)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--vllm_max_model_len", type=int, default=4096)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--wandb_project", type=str, default="biolatent-grpo-text")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--wandb_dir", type=str, default=None)

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
        "--use_reward_answer_tag_output_cot_scaled",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help=(
            "Include an output-CoT-scaled variant of `format_reward_answer_tag`. "
            "scaled_reward = (output_cot_len + 400) * base_reward / 400.0."
        ),
    )
    parser.add_argument(
        "--use_reward_answer_type_validity_output_cot_scaled",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help=(
            "Include an output-CoT-scaled variant of `reward_answer_type_validity`. "
            "scaled_reward = (output_cot_len + 400) * base_reward / 400.0."
        ),
    )
    parser.add_argument(
        "--use_reward_answer_correctness_bench_output_cot_scaled",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help=(
            "Include an output-CoT-scaled variant of `reward_answer_correctness_bench`. "
            "scaled_reward = (output_cot_len + 400) * base_reward / 400.0."
        ),
    )
    parser.add_argument(
        "--use_reward_answer_correctness",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Include legacy `reward_answer_correctness` in reward functions (non-benchmark routing).",
    )

    args = parser.parse_args()

    qwen_model_path = _resolve_qwen_model_path(args.qwen_size)
    set_tokenizer_model_path(qwen_model_path)

    argv = sys.argv[1:]
    seed_was_explicitly_provided = _arg_was_explicitly_provided(argv, "--seed")
    if seed_was_explicitly_provided:
        hf_set_seed(args.seed)

    run_name = args.run_name or f"grpo-text-{datetime.now().strftime('%m%d-%H%M')}"
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

    logger.info("Loading base text model from %s", qwen_model_path)
    model, tokenizer = load_qwen3_text_model(
        qwen_model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model = _prepare_text_lora_model(
        model,
        lora_path=args.lora_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    train_dataset = load_grpo_data(args.data_path)
    if "input_smiles" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns(["input_smiles"])

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
        vllm_ckpt=(args.vllm_ckpt or qwen_model_path),
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
    if bool(args.use_reward_answer_tag_output_cot_scaled):
        reward_funcs.append(reward_answer_tag_output_cot_scaled)
    if bool(args.use_reward_answer_type_validity_output_cot_scaled):
        reward_funcs.append(reward_answer_type_validity_output_cot_scaled)
    if bool(args.use_reward_answer_correctness_bench_output_cot_scaled):
        reward_funcs.append(reward_answer_correctness_bench_output_cot_scaled)
    if not reward_funcs:
        raise ValueError("No reward functions selected. Set at least one `--use_reward_* true` flag.")

    trainer = TextGRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[TerminalRewardPlotCallback()],
        log_reward_trace=bool(args.log_reward_trace),
        reward_trace_dir=args.reward_trace_dir,
    )

    resume = args.resume_from_checkpoint
    if resume:
        resume_s = str(resume).strip()
        if resume_s.lower() == "latest":
            ckpt = _find_latest_checkpoint(grpo_args.output_dir)
            if ckpt is None:
                raise ValueError(
                    f"--resume_from_checkpoint latest: no checkpoint-* directory found under {grpo_args.output_dir}"
                )
            resume_s = ckpt
        if not os.path.isdir(resume_s):
            raise ValueError(f"--resume_from_checkpoint path does not exist or is not a directory: {resume_s}")
        logger.info("Resuming training from checkpoint: %s", resume_s)
        trainer.train(resume_from_checkpoint=resume_s)
    else:
        trainer.train()

    final_dir = grpo_args.output_dir
    lora_dir = os.path.join(final_dir, "lora_weights")
    os.makedirs(lora_dir, exist_ok=True)
    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    logger.info("Saved LoRA to %s and tokenizer to %s", lora_dir, final_dir)


if __name__ == "__main__":
    main()
