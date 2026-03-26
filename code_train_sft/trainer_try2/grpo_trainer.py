"""
Thin adapter around TRL's official GRPOTrainer to support Bio-LatentCOT's molecule-conditioned model.

We keep TRL's implementation intact by:
- importing the installed `trl` package
- subclassing `trl.trainer.grpo_trainer.GRPOTrainer`
- overriding only the parts that must be molecule-aware:
  * generation (needs fused prompt embeddings from `model_stage3.Qwen3MoleculeLLM.get_prompt_embeddings`)
  * per-token log-prob extraction (model prepends molecule tokens, so logits must be indexed with a prefix offset)
  * passing `smiles` through the buffered batches so `compute_loss` can run
"""

from __future__ import annotations

import json
import math
import os
import time
import hashlib
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
from torch import nn
from transformers import Trainer as _HFTrainer

from trainer_try2.reward_func import is_correct_answer_bench
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_trainer import (
    GRPOTrainer as _TRL_GRPOTrainer,
    apply_chat_template,
    gather,
    is_conversational,
    logger as _trl_logger,
    profiling_context,
)
from trl.trainer.utils import entropy_from_logits, selective_log_softmax


def _normalize_smiles_batch(smiles_batch: list[Any] | None) -> list[list[str]] | None:
    if smiles_batch is None:
        return None
    out: list[list[str]] = []
    for item in smiles_batch:
        if item is None:
            out.append([])
        elif isinstance(item, str):
            out.append([item])
        else:
            out.append(list(item))
    return out

def _stable_hash01(text: str, seed: int, step: int) -> float:
    payload = f"{seed}:{step}:{text}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    val = int.from_bytes(digest[:8], "little", signed=False)
    return val / float(2**64)


class QwenMoleculeGRPOTrainer(_TRL_GRPOTrainer):
    """
    GRPOTrainer adapter for `model_stage3.Qwen3MoleculeLLM`.

    Requirements on the model:
    - `forward(..., smiles=List[List[str]])` must be supported
    - `get_prompt_embeddings(smiles_list, input_ids, attention_mask, ...) -> (prompt_embeds, prompt_attn_mask)` must exist
    - If `beta != 0`, LoRA/PEFT is expected on the *inner* text model, exposing `.disable_adapter()`.
    """

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
        training_stage: int = 3,
        corrupt_prob: float = 0.0,
        corrupt_latent_noise_std: float = 0.0,
        log_reward_trace: bool = False,
        reward_trace_dir: str | None = None,
        vllm_seed: int | None = None,
    ):
        # We implement our own vLLM prompt_embeds path for this multimodal model.
        # TRL's stock vLLM integration generates from token ids and cannot consume `prompt_embeds` built from SMILES.
        self._use_vllm_for_generation = bool(getattr(args, "use_vllm", False)) if args is not None else False
        _orig_use_vllm = None
        if args is not None and self._use_vllm_for_generation:
            _orig_use_vllm = args.use_vllm
            args.use_vllm = False

        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "disable_adapter") and not hasattr(model, "disable_adapter"):
            # Make TRL's `with unwrap_model(model).disable_adapter():` work for a wrapper model.
            model.disable_adapter = inner.disable_adapter  # type: ignore[attr-defined]

        orig_beta = float(getattr(args, "beta", 0.0) or 0.0) if args is not None else 0.0
        can_disable_adapter = hasattr(model, "disable_adapter")

        # Avoid allocating a separate reference model for wrapper models when we can disable the inner adapter instead.
        if args is not None and orig_beta != 0.0 and can_disable_adapter:
            args.beta = 0.0
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
            args.beta = orig_beta
            self.beta = orig_beta
        else:
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
            raise NotImplementedError("Tools/tool-calling is not supported in this molecule GRPO trainer.")

        self._current_smiles: list[list[str]] | None = None
        self._current_corrupt_task_latents: list[bool] | None = None
        self._current_task_latent_count: list[int] | None = None
        self.llm = None
        self.tp_group = None
        self._last_loaded_step = -1

        self.training_stage = int(training_stage)
        self.corrupt_prob = float(corrupt_prob)
        self.corrupt_latent_noise_std = float(corrupt_latent_noise_std)
        self.vllm_seed = None if vllm_seed is None else int(vllm_seed)
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

        if self._use_vllm_for_generation:
            self._init_vllm_engine(model=model, args=args)
            if args is not None and _orig_use_vllm is not None:
                args.use_vllm = _orig_use_vllm

    def _init_vllm_engine(self, model, args) -> None:
        if args is None:
            raise ValueError("use_vllm=True requires a GRPOConfig/TrainingArguments instance.")

        vllm_mode = getattr(args, "vllm_mode", "colocate")
        if vllm_mode != "colocate":
            raise NotImplementedError(
                "This smiles-based vLLM integration only supports `vllm_mode='colocate'` because it relies on "
                "prompt_embeds."
            )

        try:
            from vllm import LLM  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("use_vllm=True requires vLLM to be installed.") from e

        # Determine the HF checkpoint directory for vLLM.
        vllm_ckpt = getattr(args, "vllm_ckpt", None) or None
        if isinstance(vllm_ckpt, str) and vllm_ckpt.strip() == "":
            vllm_ckpt = None
        if vllm_ckpt is None:
            vllm_ckpt = getattr(getattr(model, "config", None), "_name_or_path", None) or getattr(
                getattr(getattr(model, "model", None), "config", None), "_name_or_path", None
            )
        if not vllm_ckpt:
            raise ValueError(
                "Could not infer a vLLM checkpoint path. Please pass `--vllm_ckpt /path/to/qwen` "
                "(a directory with config.json)."
            )

        self.vllm_mode = vllm_mode
        self.vllm_tensor_parallel_size = int(getattr(args, "vllm_tensor_parallel_size", 1) or 1)
        self.vllm_gpu_memory_utilization = float(getattr(args, "vllm_gpu_memory_utilization", 0.9))
        self.vllm_enable_sleep_mode = bool(getattr(args, "vllm_enable_sleep_mode", False))
        self.vllm_max_model_length = int(getattr(args, "vllm_max_model_length", 4096) or 4096)

        if self.vllm_tensor_parallel_size > 1:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("vllm_tensor_parallel_size > 1 requires torch.distributed to be initialized.")
            if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                raise ValueError(
                    f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                    f"({self.accelerator.num_processes}) evenly."
                )
            self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                [
                    list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                    for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                ]
            )

        # vLLM relies on these env vars for distributed execution.
        os.environ["RANK"] = str(self.accelerator.process_index)
        os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
        os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)

        # Capacity: number of sequences vLLM can keep in-flight.
        steps_per_generation = int(getattr(args, "steps_per_generation", 1) or 1)
        vllm_max_num_seqs = int(self.num_generations * args.per_device_train_batch_size * self.vllm_tensor_parallel_size * steps_per_generation)

        self.llm = LLM(
            model=vllm_ckpt,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            swap_space=0,
            max_num_seqs=vllm_max_num_seqs,
            max_model_len=self.vllm_max_model_length,
            distributed_executor_backend="external_launcher",
            seed=(
                self.vllm_seed
                if self.vllm_seed is not None
                else self.accelerator.process_index // self.vllm_tensor_parallel_size
            ),
            enable_prompt_embeds=True,
        )
        if self.vllm_enable_sleep_mode:
            self.llm.sleep(level=1)

        self.accelerator.wait_for_everyone()

    def _get_vllm_driver_model(self):
        if self.llm is None:
            raise RuntimeError("vLLM engine is not initialized.")
        # vLLM internal API differs across versions; try a few known paths.
        for path in (("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),):
            obj = self.llm
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                return obj
        raise RuntimeError("Unable to locate vLLM model object for weight loading (unsupported vLLM version).")

    def _move_model_to_vllm(self) -> None:
        if not self._use_vllm_for_generation:
            return

        # Only sync the underlying text LLM (LoRA) to vLLM; molecule fusion happens on the HF side via prompt_embeds.
        wrapper = self.accelerator.unwrap_model(self.model)
        llm_module = getattr(wrapper, "model", wrapper)

        # DeepSpeed ZeRO-3 needs parameter gathering.
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:  # pragma: no cover
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        try:
            from accelerate.utils import is_peft_model  # type: ignore
        except Exception:  # pragma: no cover
            def is_peft_model(_m):  # type: ignore
                return False

        llm_model = self._get_vllm_driver_model()

        if is_peft_model(llm_module):
            with gather_if_zero3(list(llm_module.parameters())):
                if hasattr(llm_module, "merge_adapter"):
                    llm_module.merge_adapter()

                for name, param in llm_module.named_parameters():
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if getattr(llm_module, "prefix", None) and llm_module.prefix in name:
                        continue
                    if "original_module" in name:
                        continue
                    name = name.replace("_checkpoint_wrapped_module.", "").replace("modules_to_save.default.", "")
                    with gather_if_zero3([param]):
                        llm_model.load_weights([(name, param.data)])

                if hasattr(llm_module, "unmerge_adapter"):
                    llm_module.unmerge_adapter()
        else:
            for name, param in llm_module.named_parameters():
                name = name.replace("_checkpoint_wrapped_module.", "")
                with gather_if_zero3([param]):
                    llm_model.load_weights([(name, param.data)])


    def _extract_smiles_from_inputs(self, inputs: list[dict[str, Any]]) -> list[list[str]]:
        if not inputs:
            return []
        if "smiles" in inputs[0]:
            smiles = [ex.get("smiles") for ex in inputs]
        else:
            smiles = [ex.get("input_smiles") for ex in inputs]
        smiles_norm = _normalize_smiles_batch(smiles)
        if smiles_norm is None:
            raise ValueError("Batch is missing `input_smiles` (or `smiles`) required by the molecule model.")
        return smiles_norm

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

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]):
        self._current_smiles = self._extract_smiles_from_inputs(inputs)
        self._current_task_latent_count = None

        if self.training_stage in (4, 5):
            step = int(getattr(self.state, "global_step", 0) or 0)
            seed = int(getattr(self.args, "seed", 0) or 0) if self.args is not None else 0
            flags: list[bool] = []
            for ex, sm in zip(inputs, self._current_smiles, strict=True):
                prompt = ex.get("prompt") or ""
                key = f"{prompt}\n{'.'.join(sm)}"
                r = _stable_hash01(key, seed=seed, step=step)
                flags.append(bool(r < float(self.corrupt_prob)))
            self._current_corrupt_task_latents = flags
        else:
            self._current_corrupt_task_latents = None

        output = super()._generate_and_score_completions(inputs)
        output["smiles"] = self._current_smiles

        # Stage 4: thread corruption flags + latent counts to loss computation, and mask out corrupted-but-wrong samples.
        if self.training_stage in (4, 5) and self._current_corrupt_task_latents is not None:
            device = output["prompt_ids"].device
            output["corrupt_task_latents"] = torch.tensor(
                self._current_corrupt_task_latents, device=device, dtype=torch.bool
            )
            counts = self._current_task_latent_count or [0 for _ in self._current_corrupt_task_latents]
            output["task_latent_count"] = torch.tensor(counts, device=device, dtype=torch.long)

            # Mask: keep all uncorrupted samples; for corrupted samples, only keep those that are correct.
            prompts_text = [ex.get("prompt") or "" for ex in inputs]
            labels_text = [ex.get("label") or ex.get("labels") or "" for ex in inputs]
            completions_text = self.processing_class.batch_decode(
                output["completion_ids"], skip_special_tokens=True
            )
            tasks_text = [ex.get("task") for ex in inputs]
            subtasks_text = [ex.get("subtask") for ex in inputs]
            meta_text = [ex.get("meta") for ex in inputs]
            is_correct = []
            for p, c, y, t, st, m in zip(
                prompts_text, completions_text, labels_text, tasks_text, subtasks_text, meta_text, strict=True
            ):
                is_correct.append(is_correct_answer_bench(p, c, y, task=t, subtask=st, meta=m))
            loss_mask = torch.tensor(
                [((not corr) or ok) for corr, ok in zip(self._current_corrupt_task_latents, is_correct, strict=True)],
                device=device,
                dtype=output["completion_mask"].dtype,
            )
            output["completion_mask"] = output["completion_mask"] * loss_mask.unsqueeze(1)

        return output

    def _generate_single_turn(self, prompts: list):
        if prompts and not isinstance(prompts[0], str):
            raise NotImplementedError("Conversational prompts are not supported for the molecule GRPO trainer.")

        generate_inputs = self.processing_class(text=prompts, padding=True, padding_side="left", return_tensors="pt")
        # Important: GRPOTrainer overrides `_prepare_inputs` with buffering logic; for generation-time tokenization we
        # only need the vanilla HF `Trainer._prepare_inputs` (device placement, etc.).
        generate_inputs = _HFTrainer._prepare_inputs(self, generate_inputs)
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]

        smiles = self._current_smiles
        if smiles is None or len(smiles) != prompt_ids.size(0):
            raise RuntimeError(
                "Internal error: missing/size-mismatched smiles for generation. "
                f"smiles={None if smiles is None else len(smiles)}, batch={prompt_ids.size(0)}."
            )

        # vLLM path: build prompt_embeds (HF side), then generate token ids using vLLM.
        if self._use_vllm_for_generation:
            if self.llm is None:
                raise RuntimeError("use_vllm=True but vLLM engine is not initialized.")

            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            wrapper = self.accelerator.unwrap_model(self.model)
            if not hasattr(wrapper, "get_prompt_embeddings"):
                raise AttributeError(
                    "Model must implement `get_prompt_embeddings(smiles_list, input_ids, attention_mask, ...)` "
                    "to support vLLM prompt_embeds generation."
                )

            all_prompt_ids = prompt_ids
            all_prompt_mask = prompt_mask
            all_smiles = smiles
            all_corrupt = self._current_corrupt_task_latents
            orig_size = len(prompts)
            if self.vllm_tensor_parallel_size > 1:
                if self.tp_group is None:
                    raise RuntimeError("TP group not initialized.")
                gathered_prompt_ids = [torch.empty_like(prompt_ids) for _ in range(self.vllm_tensor_parallel_size)]
                gathered_prompt_mask = [torch.empty_like(prompt_mask) for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather(gathered_prompt_ids, prompt_ids, group=self.tp_group)
                torch.distributed.all_gather(gathered_prompt_mask, prompt_mask, group=self.tp_group)

                gathered_smiles = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_smiles, smiles, group=self.tp_group)
                all_smiles = [s for sub in gathered_smiles for s in sub]

                if self.training_stage in (4, 5):
                    gathered_corrupt = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_corrupt, self._current_corrupt_task_latents, group=self.tp_group)
                    all_corrupt = [bool(x) for sub in gathered_corrupt for x in (sub or [])]

                all_prompt_ids = torch.cat(gathered_prompt_ids, dim=0)
                all_prompt_mask = torch.cat(gathered_prompt_mask, dim=0)

            if self.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up(level=1)

            with torch.inference_mode():
                prompt_embeds, prompt_attn = wrapper.get_prompt_embeddings(
                    smiles_list=all_smiles,
                    input_ids=all_prompt_ids,
                    attention_mask=all_prompt_mask,
                    refine_bio_tokens=True,
                    corrupt_task_latents=all_corrupt,
                    corrupt_task_latent_noise_std=float(self.corrupt_latent_noise_std),
                )
            embed_list = [prompt_embeds[i][prompt_attn[i].bool()].contiguous() for i in range(prompt_embeds.size(0))]
            latent_counts_all = list(getattr(wrapper, "_last_task_latent_counts", []) or [])

            from vllm import SamplingParams  # type: ignore

            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=0.0 if self.min_p is None else self.min_p,
                max_tokens=self.max_completion_length,
                logprobs=0,
            )

            start = time.time()
            all_outputs = self.llm.generate(
                [{"prompt_embeds": e} for e in embed_list],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            if self.accelerator.is_main_process:
                _ = time.time() - start

            all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
            if self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids_list = all_completion_ids[tp_slice]
                latent_counts = latent_counts_all[tp_slice] if latent_counts_all else []
            else:
                completion_ids_list = all_completion_ids
                latent_counts = latent_counts_all

            prompt_ids_list = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            logprobs = None
            if self.training_stage in (4, 5):
                self._current_task_latent_count = [int(x) for x in (latent_counts or [0 for _ in range(orig_size)])]
                extra_fields = {
                    "corrupt_task_latents": list(self._current_corrupt_task_latents or [False for _ in range(orig_size)]),
                    "task_latent_count": list(self._current_task_latent_count),
                }
            else:
                extra_fields = {}
            return prompt_ids_list, completion_ids_list, logprobs, extra_fields

        # HF path: prompt_embeds + transformers generate
        with (
            unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,
            ) as unwrapped_model,
            torch.no_grad(),
        ):
            if not hasattr(unwrapped_model, "get_prompt_embeddings"):
                raise AttributeError(
                    "Model must implement `get_prompt_embeddings(smiles_list, input_ids, attention_mask, ...)` "
                    "for molecule-conditioned GRPO generation."
                )

            prompt_embeds, prompt_attn_mask = unwrapped_model.get_prompt_embeddings(
                smiles_list=smiles,
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                refine_bio_tokens=True,
                corrupt_task_latents=self._current_corrupt_task_latents,
                corrupt_task_latent_noise_std=float(self.corrupt_latent_noise_std),
            )
            fused_prompt_len = prompt_embeds.size(1)
            latent_counts = list(getattr(unwrapped_model, "_last_task_latent_counts", []) or [])

            out = unwrapped_model.model.generate(
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_attn_mask,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = out.sequences  # (B, ?)
        gen_len = len(out.scores)
        if sequences.size(1) == gen_len:
            completion_ids = sequences
        elif sequences.size(1) == fused_prompt_len + gen_len:
            completion_ids = sequences[:, fused_prompt_len:]
        else:
            # Fallback: keep the generated tail.
            completion_ids = sequences[:, -gen_len:] if gen_len > 0 else sequences[:, 0:0]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        if completion_ids.size(1) > 0:
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
            has_eos = is_eos.any(dim=1)
            eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
            sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        else:
            completion_mask = torch.zeros_like(completion_ids, dtype=torch.int)

        prompt_ids_list = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
        completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
        logprobs = None
        if self.training_stage in (4, 5):
            self._current_task_latent_count = [int(x) for x in (latent_counts or [0 for _ in range(len(prompts))])]
            extra_fields = {
                "corrupt_task_latents": list(self._current_corrupt_task_latents or [False for _ in range(len(prompts))]),
                "task_latent_count": list(self._current_task_latent_count),
            }
        else:
            extra_fields = {}
        return prompt_ids_list, completion_ids_list, logprobs, extra_fields

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        smiles: Optional[list[list[str]]] = None,
    ):
        smiles_to_use = smiles if smiles is not None else self._current_smiles
        if smiles_to_use is None:
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                num_images=num_images,
                pixel_attention_mask=pixel_attention_mask,
                image_sizes=image_sizes,
                token_type_ids=token_type_ids,
            )

        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []

        model_to_check = self.accelerator.unwrap_model(model)
        # use_prompt_embeds = bool(getattr(model_to_check, "is_both_latent", False))
        use_prompt_embeds = True

        if use_prompt_embeds:
            if not hasattr(model_to_check, "get_prompt_embeddings"):
                raise AttributeError(
                    "Model with is_both_latent=True must implement `get_prompt_embeddings(...)` for GRPO logps."
                )
            if not hasattr(model_to_check, "model"):
                raise AttributeError("Wrapper model is missing `.model` (inner text model).")

            embed = model_to_check.model.get_input_embeddings()

            corrupt_flags = self._current_corrupt_task_latents
            for start in range(0, input_ids.size(0), batch_size):
                end = start + batch_size
                input_ids_batch = input_ids[start:end]
                attention_mask_batch = attention_mask[start:end]
                smiles_batch = smiles_to_use[start:end]

                prompt_ids_batch = input_ids_batch[:, :-logits_to_keep]
                prompt_mask_batch = attention_mask_batch[:, :-logits_to_keep]
                completion_ids = input_ids_batch[:, -logits_to_keep:]
                completion_attn = attention_mask_batch[:, -logits_to_keep:]

                corrupt_batch = None
                if corrupt_flags is not None:
                    corrupt_batch = corrupt_flags[start:end]

                prompt_embeds, prompt_attn = model_to_check.get_prompt_embeddings(
                    smiles_list=smiles_batch,
                    input_ids=prompt_ids_batch,
                    attention_mask=prompt_mask_batch,
                    refine_bio_tokens=True,
                    corrupt_task_latents=corrupt_batch,
                    corrupt_task_latent_noise_std=float(self.corrupt_latent_noise_std),
                )
                completion_embeds = embed(completion_ids).to(dtype=prompt_embeds.dtype)
                full_embeds = torch.cat([prompt_embeds, completion_embeds], dim=1)
                full_mask = torch.cat([prompt_attn, completion_attn], dim=1)

                out = model_to_check.model(
                    inputs_embeds=full_embeds,
                    attention_mask=full_mask,
                    return_dict=True,
                    use_cache=False,
                )
                logits_full = out.logits[:, :-1, :]
                prefix_len = prompt_embeds.size(1)
                start_pos = max(prefix_len - 1, 0)
                logits = logits_full[:, start_pos : start_pos + logits_to_keep, :]
                logits = logits / self.temperature

                logps = selective_log_softmax(logits, completion_ids) * completion_attn.to(torch.float32)
                all_logps.append(logps)

                if compute_entropy:
                    with torch.no_grad():
                        entropies = entropy_from_logits(logits) * completion_attn.to(torch.float32)
                    all_entropies.append(entropies)

            logps = torch.cat(all_logps, dim=0)
            entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
            return logps, entropies

        # num_queries = int(getattr(model_to_check, "num_queries", 0))
        # if num_queries <= 0:
        #     # Unexpected: fall back to TRL behavior.
        #     return super()._get_per_token_logps_and_entropies(
        #         model,
        #         input_ids,
        #         attention_mask,
        #         logits_to_keep,
        #         batch_size=batch_size,
        #         compute_entropy=compute_entropy,
        #         pixel_values=pixel_values,
        #         image_grid_thw=image_grid_thw,
        #         num_images=num_images,
        #         pixel_attention_mask=pixel_attention_mask,
        #         image_sizes=image_sizes,
        #         token_type_ids=token_type_ids,
        #     )

        # for start in range(0, input_ids.size(0), batch_size):
        #     end = start + batch_size
        #     input_ids_batch = input_ids[start:end]
        #     attention_mask_batch = attention_mask[start:end]
        #     smiles_batch = smiles_to_use[start:end]

        #     model_inputs = {
        #         "input_ids": input_ids_batch,
        #         "attention_mask": attention_mask_batch,
        #         "use_cache": False,
        #         "smiles": smiles_batch,
        #     }
        #     logits_full = model(**model_inputs).logits  # (B, L_fused, V)
        #     logits_full = logits_full[:, :-1, :]  # next-token shift

        #     completion_ids = input_ids_batch[:, -logits_to_keep:]
        #     completion_attn = attention_mask_batch[:, -logits_to_keep:]

        #     completion_lens = completion_attn.sum(dim=1).to(torch.long)
        #     text_lens = attention_mask_batch.sum(dim=1).to(torch.long)
        #     prompt_lens = (text_lens - completion_lens).clamp(min=0)

        #     mol_prefix_lens = torch.tensor(
        #         [(len(s) if isinstance(s, list) else 0) * (num_queries + 2) for s in smiles_batch],
        #         device=logits_full.device,
        #         dtype=torch.long,
        #     )

        #     j = torch.arange(logits_to_keep, device=logits_full.device, dtype=torch.long).unsqueeze(0)
        #     pos = mol_prefix_lens.unsqueeze(1) + prompt_lens.unsqueeze(1) + j - 1
        #     pos = pos.clamp(min=0, max=logits_full.size(1) - 1)
        #     batch_idx = torch.arange(logits_full.size(0), device=logits_full.device, dtype=torch.long).unsqueeze(1)
        #     logits = logits_full[batch_idx, pos]  # (B, T, V)
        #     logits = logits / self.temperature

        #     logps = selective_log_softmax(logits, completion_ids) * completion_attn.to(torch.float32)
        #     all_logps.append(logps)

        #     if compute_entropy:
        #         with torch.no_grad():
        #             entropies = entropy_from_logits(logits) * completion_attn.to(torch.float32)
        #         all_entropies.append(entropies)

        # logps = torch.cat(all_logps, dim=0)
        # entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        # return logps, entropies

    def _compute_loss(self, model, inputs):
        # Ensure the current batch's `smiles` is visible to `_get_per_token_logps_and_entropies` when the parent
        # implementation calls it (it doesn't thread `smiles` explicitly).
        if "smiles" in inputs:
            self._current_smiles = inputs["smiles"]
        if "corrupt_task_latents" in inputs:
            val = inputs["corrupt_task_latents"]
            if isinstance(val, torch.Tensor):
                self._current_corrupt_task_latents = [bool(x) for x in val.detach().cpu().tolist()]
            else:
                self._current_corrupt_task_latents = [bool(x) for x in val]
        return super()._compute_loss(model, inputs)

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
