"""
Thin adapter around TRL's official GRPOTrainer for plain text-only Qwen training.

Unlike the molecule trainer, this class does not override generation or log-prob
paths. It only adds per-rollout reward trace logging while preserving TRL's
stock text-only behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

import torch
from torch import nn
from transformers import Trainer as _HFTrainer
from trl.trainer.grpo_trainer import (
    GRPOTrainer as _TRL_GRPOTrainer,
    apply_chat_template,
    gather,
    is_conversational,
    logger as _trl_logger,
    profiling_context,
)


class TextGRPOTrainer(_TRL_GRPOTrainer):
    def __init__(
        self,
        model,
        reward_funcs,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        tools=None,
        rollout_func=None,
        log_reward_trace: bool = False,
        reward_trace_dir: str | None = None,
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            tools=tools,
            rollout_func=rollout_func,
        )

        if getattr(self, "tools", None):
            raise NotImplementedError("Tools/tool-calling is not supported in this text GRPO trainer.")

        self.log_reward_trace = bool(log_reward_trace)
        self.reward_trace_dir: str | None = None
        self._reward_trace_path: str | None = None
        if self.log_reward_trace:
            if len(set(self.reward_func_names)) != len(self.reward_func_names):
                raise ValueError(
                    "log_reward_trace=True requires unique reward function names. "
                    f"Got: {self.reward_func_names}"
                )
            trace_dir = reward_trace_dir
            if trace_dir is None:
                output_dir = getattr(args, "output_dir", None) if args is not None else None
                if not output_dir:
                    raise ValueError(
                        "log_reward_trace=True requires `reward_trace_dir` or `args.output_dir` to be set."
                    )
                trace_dir = os.path.join(str(output_dir), "reward_trace")
            self.reward_trace_dir = os.path.abspath(str(trace_dir))
            os.makedirs(self.reward_trace_dir, exist_ok=True)
            self._reward_trace_path = os.path.join(
                self.reward_trace_dir,
                f"rank_{self.accelerator.process_index}.jsonl",
            )

    @staticmethod
    def _hash_text(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Expected a string to hash, got {type(text).__name__}.")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _reward_value_for_trace(value: float) -> float | None:
        value_f = float(value)
        if math.isnan(value_f):
            return None
        return value_f

    def _append_reward_trace(
        self,
        *,
        inputs: list[dict[str, Any]],
        prompts: list[str],
        completions: list[str],
        rewards_per_func: torch.Tensor,
    ) -> None:
        if not self.log_reward_trace:
            return
        if self._reward_trace_path is None:
            raise RuntimeError("Reward trace logging is enabled but trace path is not initialized.")
        if len(inputs) != len(prompts) or len(prompts) != len(completions):
            raise ValueError(
                "Reward trace logging requires inputs/prompts/completions to have identical lengths. "
                f"Got inputs={len(inputs)}, prompts={len(prompts)}, completions={len(completions)}."
            )
        if rewards_per_func.ndim != 2:
            raise ValueError(
                f"Expected rewards_per_func to be rank-2, got shape={tuple(rewards_per_func.shape)}."
            )
        if rewards_per_func.size(0) != len(prompts):
            raise ValueError(
                "Reward trace logging requires rewards_per_func rows to match completions. "
                f"Got rewards={rewards_per_func.size(0)}, completions={len(prompts)}."
            )
        if rewards_per_func.size(1) != len(self.reward_func_names):
            raise ValueError(
                "Reward trace logging requires rewards_per_func columns to match reward function names. "
                f"Got rewards={rewards_per_func.size(1)}, names={len(self.reward_func_names)}."
            )

        step = int(getattr(self.state, "global_step", 0) or 0)
        process_index = int(self.accelerator.process_index)
        rewards_cpu = rewards_per_func.detach().cpu()
        lines: list[str] = []
        for local_rollout_index, (example, prompt, completion) in enumerate(
            zip(inputs, prompts, completions, strict=True)
        ):
            if not isinstance(prompt, str):
                raise TypeError(
                    "Reward trace logging currently expects string prompts only. "
                    f"Got {type(prompt).__name__}."
                )
            if not isinstance(completion, str):
                raise TypeError(
                    "Reward trace logging currently expects string completions only. "
                    f"Got {type(completion).__name__}."
                )
            reward_values = rewards_cpu[local_rollout_index].tolist()
            reward_trace = {
                name: self._reward_value_for_trace(value)
                for name, value in zip(self.reward_func_names, reward_values, strict=True)
            }
            valid_rewards = [value for value in reward_trace.values() if value is not None]
            row = {
                "global_step": step,
                "process_index": process_index,
                "local_rollout_index": local_rollout_index,
                "task": example.get("task"),
                "subtask": example.get("subtask"),
                "reward_total_raw": (sum(valid_rewards) if valid_rewards else None),
                "subrewards": reward_trace,
                "prompt_hash": self._hash_text(prompt),
                "completion_hash": self._hash_text(completion),
            }
            lines.append(json.dumps(row, ensure_ascii=False))

        with open(self._reward_trace_path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.write("\n")

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = _HFTrainer._prepare_inputs(self, reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            _trl_logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        self._append_reward_trace(
            inputs=inputs,
            prompts=prompts,
            completions=completions,
            rewards_per_func=rewards_per_func,
        )

        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func
