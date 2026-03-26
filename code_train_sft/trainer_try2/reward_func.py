"""
Reward functions (and shared helpers) for GRPO training in Bio-LatentCOT.

This module is intentionally dependency-light so it can be imported by:
- `train_grpo_try2.py` (to build `reward_funcs`)
- `trainer_try2/grpo_trainer.py` (for stage4/5 corruption loss masking correctness checks)
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Any, Callable, List, Optional

import torch


def format_reward_answer_tag(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    corrupt_task_latents: Optional[List[bool]] = None,
    **kwargs,
):
    """
    Minimal "try1" reward:
    - reward 1.0 if model outputs a non-empty `<answer> ... </answer>` span
    - else 0.0
    """
    if corrupt_task_latents is None:
        corrupt_flags = None
    elif isinstance(corrupt_task_latents, torch.Tensor):
        corrupt_flags = [bool(x) for x in corrupt_task_latents.detach().cpu().tolist()]
    else:
        corrupt_flags = [bool(x) for x in corrupt_task_latents]

    rewards = []
    for i, c in enumerate(completions):
        if corrupt_flags is not None and i < len(corrupt_flags) and corrupt_flags[i]:
            rewards.append(0.0)
            continue
        c = c or ""
        lo = c.lower()
        start = lo.find("<answer>")
        end = lo.find("</answer>")
        if start == -1 or end == -1 or end <= start:
            rewards.append(0.0)
            continue
        try:
            inner = c[start + len("<answer>") : end].strip()
            rewards.append(1.0 if len(inner) > 0 else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)

_PROMPT_EXPECT_RE = re.compile(
    r"formatted as\s*<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL
)

_RDKit_LOGS_DISABLED = False
_OUTPUT_COT_SCALE_OFFSET = 400.0
_OUTPUT_COT_SCALE_DIVISOR = 400.0


def _get_rdkit_chem():
    global _RDKit_LOGS_DISABLED
    try:
        from rdkit import Chem  # type: ignore
        from rdkit import RDLogger  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("RDKit is required for SMILES rewards.") from e

    if not _RDKit_LOGS_DISABLED:
        RDLogger.DisableLog("rdApp.error")
        RDLogger.DisableLog("rdApp.warning")
        _RDKit_LOGS_DISABLED = True
    return Chem


def _get_runtime_tokenizer():
    try:
        from dataloader import get_tokenizer
    except Exception as e:  # pragma: no cover
        raise ImportError("Failed to import `get_tokenizer()` from `dataloader`.") from e

    tokenizer = get_tokenizer()
    if tokenizer is None:
        raise RuntimeError("Runtime tokenizer is None.")
    return tokenizer


def _extract_output_cot_prefix(completion: str) -> str:
    if not isinstance(completion, str):
        raise TypeError(f"Completion must be a string, got: {type(completion)}")
    answer_start = completion.rfind("<answer>")
    if answer_start == -1:
        return completion
    return completion[:answer_start]


def _get_output_cot_token_len(completion: str) -> int:
    prefix = _extract_output_cot_prefix(completion)
    tokenizer = _get_runtime_tokenizer()
    encoded = tokenizer(prefix, add_special_tokens=False)
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, list):
        raise TypeError(f"Tokenizer returned non-list input_ids: {type(input_ids)}")
    return len(input_ids)


def _apply_output_cot_scaling(
    base_reward_func: Callable[..., List[float]],
    *,
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    **kwargs,
) -> List[float]:
    if not isinstance(completions, list):
        raise TypeError(f"`completions` must be a list, got: {type(completions)}")

    base_rewards = base_reward_func(
        prompts=prompts,
        completions=completions,
        completion_ids=completion_ids,
        **kwargs,
    )
    if isinstance(base_rewards, torch.Tensor):
        base_rewards = base_rewards.detach().cpu().tolist()
    if not isinstance(base_rewards, list):
        raise TypeError(f"Base reward func must return a list, got: {type(base_rewards)}")
    if len(base_rewards) != len(completions):
        raise ValueError(
            f"Base reward length mismatch: got {len(base_rewards)} rewards for {len(completions)} completions."
        )

    scaled_rewards: list[float] = []
    for completion, reward in zip(completions, base_rewards, strict=True):
        output_cot_len = float(_get_output_cot_token_len(completion))
        scaled_rewards.append(((output_cot_len + _OUTPUT_COT_SCALE_OFFSET) * float(reward)) / _OUTPUT_COT_SCALE_DIVISOR)
    return scaled_rewards


def _infer_expected_answer_type(prompt: str) -> str:
    """
    Infer the expected answer type from the prompt formatting instruction inserted by `extract_fields()`.

    Returns: one of {"smiles", "number", "yesno", "unknown"}.
    """
    m = _PROMPT_EXPECT_RE.search(prompt or "")
    if not m:
        return "unknown"
    spec = (m.group(1) or "").strip().lower()
    if "smiles" in spec:
        return "smiles"
    if "yes" in spec and "no" in spec:
        return "yesno"
    if "number" in spec:
        return "number"
    return "unknown"


def _extract_answer_text(completion: str) -> Optional[str]:
    m = _ANSWER_RE.search(completion or "")
    if not m:
        return None
    return m.group(1).strip()


def _canon_smiles(smiles: str) -> Optional[str]:
    Chem = _get_rdkit_chem()
    try:
        mol = Chem.MolFromSmiles(smiles)  # type: ignore[attr-defined]
    except Exception:
        mol = None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)  # type: ignore[attr-defined]
    except Exception:
        return None


def _parse_meta(meta: Any) -> dict[str, Any]:
    if meta is None:
        raise ValueError("meta is None (expected ChemCoTBench meta JSON).")
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        if not meta.strip():
            raise ValueError("meta is an empty string (expected ChemCoTBench meta JSON).")
        # Let JSON parsing errors surface; don't silently swallow malformed meta.
        return json.loads(meta)
    raise TypeError(f"Unsupported meta type: {type(meta)}")


def reward_answer_type_validity(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    corrupt_task_latents: Optional[List[bool]] = None,
    **kwargs,
):
    """
    Reward 0.5 if the `<answer>...</answer>` content matches the *type* the prompt asks for:
    - SMILES: RDKit parses
    - Number: parses as float
    - Yes/No: matches yes/no (case-insensitive)
    Otherwise reward 0.0.
    """
    if corrupt_task_latents is None:
        corrupt_flags = None
    elif isinstance(corrupt_task_latents, torch.Tensor):
        corrupt_flags = [bool(x) for x in corrupt_task_latents.detach().cpu().tolist()]
    else:
        corrupt_flags = [bool(x) for x in corrupt_task_latents]

    rewards: list[float] = []
    for i, (p, c) in enumerate(zip(prompts, completions)):
        if corrupt_flags is not None and i < len(corrupt_flags) and corrupt_flags[i]:
            rewards.append(0.0)
            continue
        expected = _infer_expected_answer_type(p)
        answer = _extract_answer_text(c or "")
        if not answer:
            rewards.append(0.0)
            continue

        cleaned = answer.strip().strip("\"'`")
        if expected == "smiles":
            Chem = _get_rdkit_chem()

            cleaned = re.sub(r"^\s*smiles\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE).strip()
            candidates = [cleaned]
            if re.search(r"\s", cleaned):
                candidates.append(cleaned.split()[0])

            ok = False
            for cand in candidates:
                try:
                    mol = Chem.MolFromSmiles(cand)  # type: ignore[attr-defined]
                except Exception:
                    mol = None
                if mol is not None:
                    ok = True
                    break
            rewards.append(0.5 if ok else 0.0)
        elif expected == "number":
            cleaned = re.sub(r"^\s*(count|number)\s*[:=]\s*", "", cleaned, flags=re.IGNORECASE).strip()
            cleaned = cleaned.replace(",", "")
            try:
                float(cleaned)
                rewards.append(0.5)
            except Exception:
                rewards.append(0.0)
        elif expected == "yesno":
            lo = cleaned.lower()
            lo = re.sub(r"^\s*(answer|output)\s*[:=]\s*", "", lo).strip()
            if lo in {"yes", "no"}:
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)

    return rewards


def reward_answer_tag_output_cot_scaled(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    **kwargs,
):
    return _apply_output_cot_scaling(
        format_reward_answer_tag,
        prompts=prompts,
        completions=completions,
        completion_ids=completion_ids,
        **kwargs,
    )


def reward_answer_type_validity_output_cot_scaled(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    **kwargs,
):
    return _apply_output_cot_scaling(
        reward_answer_type_validity,
        prompts=prompts,
        completions=completions,
        completion_ids=completion_ids,
        **kwargs,
    )


def reward_answer_correctness(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    label: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    **kwargs,
):
    """
    Reward 1.0 if the extracted `<answer>...</answer>` matches the ground-truth label.

    Requires `load_grpo_data()` to keep the `label` column.
    """
    gt_list = labels if labels is not None else label
    if gt_list is None:
        # If labels are not present in the dataset, this reward is disabled.
        return [0.0 for _ in completions]

    rewards: list[float] = []
    for p, c, gt in zip(prompts, completions, gt_list):
        expected = _infer_expected_answer_type(p)
        pred = _extract_answer_text(c or "")
        gold = _extract_answer_text(gt or "") or (gt or "").strip()
        if pred is None:
            rewards.append(0.0)
            continue

        pred_clean = pred.strip().strip("\"'`")
        gold_clean = (gold or "").strip().strip("\"'`")

        if expected == "smiles":
            Chem = _get_rdkit_chem()

            def canon(s: str) -> Optional[str]:
                try:
                    mol = Chem.MolFromSmiles(s)  # type: ignore[attr-defined]
                except Exception:
                    mol = None
                if mol is None:
                    return None
                try:
                    return Chem.MolToSmiles(mol, canonical=True)  # type: ignore[attr-defined]
                except Exception:
                    return None

            pred_can = canon(pred_clean)
            gold_can = canon(gold_clean)
            rewards.append(1.0 if (pred_can is not None and gold_can is not None and pred_can == gold_can) else 0.0)
        elif expected == "number":
            def parse_num(s: str) -> Optional[float]:
                try:
                    return float(s)
                except Exception:
                    return None

            pn = parse_num(pred_clean)
            gn = parse_num(gold_clean)
            if pn is None or gn is None:
                rewards.append(0.0)
            else:
                rewards.append(1.0 if math.isclose(pn, gn, rel_tol=1e-3, abs_tol=1e-3) else 0.0)
        elif expected == "yesno":
            def norm_yesno(s: str) -> Optional[str]:
                s = s.strip().lower()
                if s in {"yes"}:
                    return "yes"
                if s in {"no"}:
                    return "no"
                return None

            py = norm_yesno(pred_clean)
            gy = norm_yesno(gold_clean)
            rewards.append(1.0 if (py is not None and gy is not None and py == gy) else 0.0)
        else:
            rewards.append(1.0 if pred_clean.strip().lower() == gold_clean.strip().lower() and gold_clean != "" else 0.0)

    return rewards


_BENCH_REWARD_UTILS_ON_PATH = False
_MOLOPT_EVALUATER_CACHE: dict[str, object] = {}


def _ensure_bench_reward_utils_importable() -> None:
    global _BENCH_REWARD_UTILS_ON_PATH
    if _BENCH_REWARD_UTILS_ON_PATH:
        return
    # This file lives in `code_train_sft/trainer_try2/`; reward_utils is a sibling folder of `trainer_try2/`.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reward_utils_dir = os.path.join(os.path.dirname(current_dir), "reward_utils")
    if reward_utils_dir not in sys.path:
        sys.path.append(reward_utils_dir)
    _BENCH_REWARD_UTILS_ON_PATH = True


def _clean_group_name(group: Optional[str]) -> Optional[str]:
    if group is None:
        return None
    # Some benchmark meta strings include a trailing '.' (e.g., "carboxyl.")
    return str(group).strip().rstrip(".").strip()


def _extract_smiles_candidate(answer_text: str) -> str:
    s = (answer_text or "").strip().strip("\"'`")
    s = re.sub(r"^\s*smiles\s*[:=]\s*", "", s, flags=re.IGNORECASE).strip()
    if re.search(r"\s", s):
        s = s.split()[0]
    return s.strip().rstrip(".").strip()


def is_correct_answer_bench(
    prompt: str,
    completion: str,
    label: str = "",
    *,
    task: Optional[str] = None,
    subtask: Optional[str] = None,
    meta: Any = None,
) -> bool:
    """
    Single-sample correctness check aligned with `reward_answer_correctness_bench`.

    This is used by the GRPO trainer to build stage4/5 corruption loss masks.
    """
    expected = _infer_expected_answer_type(prompt or "")
    pred = _extract_answer_text(completion or "")
    if pred is None:
        return False

    if expected == "smiles":
        pred_clean = _extract_smiles_candidate(pred)

        # mol_edit: functional-group constraints (no fixed target SMILES).
        if task == "mol_edit":
            meta_dict = _parse_meta(meta)
            src = meta_dict.get("molecule")
            if not (isinstance(src, str) and src.strip()):
                return False

            _ensure_bench_reward_utils_importable()
            from ChemCoTBench.eval_moledit import (  # type: ignore
                check_edit_add_valid,
                check_edit_del_valid,
                check_edit_sub_valid,
            )

            if subtask == "add":
                group = _clean_group_name(meta_dict.get("added_group"))
                if not group:
                    return False
                return bool(check_edit_add_valid(src=src, tgt=pred_clean, group=group))
            if subtask == "delete":
                group = _clean_group_name(meta_dict.get("removed_group"))
                if not group:
                    return False
                return bool(check_edit_del_valid(src=src, tgt=pred_clean, group=group))
            if subtask == "sub":
                add_group = _clean_group_name(meta_dict.get("added_group"))
                remove_group = _clean_group_name(meta_dict.get("removed_group"))
                if not add_group or not remove_group:
                    return False
                return bool(
                    check_edit_sub_valid(src=src, tgt=pred_clean, remove_group=remove_group, add_group=add_group)
                )
            return False

        # mol_opt: property improvement (oracle-based).
        if task == "mol_opt":
            meta_dict = _parse_meta(meta)
            src = meta_dict.get("molecule")
            prop_dict = {
                "logp": "logp",
                "solubility": "solubility",
                "qed": "qed",
                "drd": "drd2",
                "jnk": "jnk3",
                "gsk": "gsk3b",
            }
            prop = prop_dict.get(str(subtask or "").strip().lower())
            if not isinstance(src, str) or not src.strip():
                raise ValueError(f"mol_opt meta missing valid `molecule`: {src!r}")
            if not prop:
                raise ValueError(f"Unknown mol_opt subtask: {subtask!r}")

            _ensure_bench_reward_utils_importable()
            from ChemCoTBench.core.eval_metric import mol_opt_evaluater  # type: ignore

            evaluater = _MOLOPT_EVALUATER_CACHE.get(prop)
            if evaluater is None:
                evaluater = mol_opt_evaluater(prop=prop)
                _MOLOPT_EVALUATER_CACHE[prop] = evaluater

            oracle = getattr(evaluater, "property_oracle", None)
            if oracle is None:
                raise AttributeError(f"mol_opt_evaluater(prop={prop!r}) missing `property_oracle`.")

            # Treat invalid predicted SMILES as incorrect (no reward), but do not silently swallow other errors.
            if _canon_smiles(src) is None or _canon_smiles(pred_clean) is None:
                return False

            src_score = oracle(src)
            tgt_score = oracle(pred_clean)
            if src_score is None or tgt_score is None:
                return False
            return bool((tgt_score - src_score) > 0)

        # Default SMILES: exact match (canonicalized).
        gold = _extract_answer_text(label or "") or (label or "").strip()
        gold_clean = _extract_smiles_candidate(gold)
        pred_can = _canon_smiles(pred_clean)
        gold_can = _canon_smiles(gold_clean)
        return bool(pred_can is not None and gold_can is not None and pred_can == gold_can)

    # Non-SMILES: strict matching against label.
    gold = _extract_answer_text(label or "") or (label or "").strip()
    pred_clean = pred.strip().strip("\"'`")
    gold_clean = (gold or "").strip().strip("\"'`")

    if expected == "number":
        num_re = re.compile(r"^\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$")
        p_s = pred_clean.replace(",", "").strip()
        g_s = gold_clean.replace(",", "").strip()
        if not num_re.match(p_s) or not num_re.match(g_s):
            return False
        pn = float(p_s)
        gn = float(g_s)
        return bool(math.isclose(pn, gn, rel_tol=1e-3, abs_tol=1e-3))

    if expected == "yesno":
        lo_p = pred_clean.strip().lower()
        lo_g = gold_clean.strip().lower()
        if lo_p not in {"yes", "no"} or lo_g not in {"yes", "no"}:
            return False
        return lo_p == lo_g

    return bool(pred_clean.strip().lower() == gold_clean.strip().lower() and gold_clean != "")


def reward_answer_correctness_bench(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    label: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    corrupt_task_latents: Optional[List[bool]] = None,
    task: Optional[List[str]] = None,
    subtask: Optional[List[str]] = None,
    meta: Optional[List[str]] = None,
    **kwargs,
):
    """
    Benchmark-aware correctness reward for ChemCoTBench-style data.

    - For numeric and yes/no answers: keep the current strict matching logic.
    - For SMILES answers:
        - mol_edit: check functional-group edit validity (no fixed target SMILES)
        - mol_opt: check property improvement using benchmark oracle
        - mol_und/reaction: exact match (as before)

    Returns:
      2.0 if correct, else 0.0
    """
    gt_list = labels if labels is not None else label

    if corrupt_task_latents is None:
        corrupt_flags = None
    elif isinstance(corrupt_task_latents, torch.Tensor):
        corrupt_flags = [bool(x) for x in corrupt_task_latents.detach().cpu().tolist()]
    else:
        corrupt_flags = [bool(x) for x in corrupt_task_latents]

    rewards: list[float] = []
    for i, (p, c) in enumerate(zip(prompts, completions, strict=True)):
        if corrupt_flags is not None and i < len(corrupt_flags) and corrupt_flags[i]:
            rewards.append(0.0)
            continue
        t = task[i] if task is not None and i < len(task) else None
        st = subtask[i] if subtask is not None and i < len(subtask) else None
        m = meta[i] if meta is not None and i < len(meta) else None
        gt = gt_list[i] if gt_list is not None and i < len(gt_list) else ""
        ok = is_correct_answer_bench(p, c, gt, task=t, subtask=st, meta=m)
        rewards.append(2.0 if ok else 0.0)

    return rewards


def reward_answer_correctness_bench_output_cot_scaled(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    **kwargs,
):
    return _apply_output_cot_scaling(
        reward_answer_correctness_bench,
        prompts=prompts,
        completions=completions,
        completion_ids=completion_ids,
        **kwargs,
    )


def reward_stage4_corrupt_or_correct(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    label: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    corrupt_task_latents: Optional[List[bool]] = None,
    **kwargs,
):
    """
    Stage 4 reward:
    - If corrupted: reward = -2 if correct else 0
    - If not corrupted: reward = correctness (2 if correct else 0)
    """
    gt_list = labels if labels is not None else label
    if gt_list is None:
        return [0.0 for _ in completions]

    if corrupt_task_latents is None:
        corrupt_task_latents = [False for _ in completions]

    correctness = reward_answer_correctness_bench(
        prompts=prompts,
        completions=completions,
        completion_ids=completion_ids,
        labels=gt_list,
        task=kwargs.get("task"),
        subtask=kwargs.get("subtask"),
        meta=kwargs.get("meta"),
    )
    out: list[float] = []
    for is_corrupt, corr in zip(corrupt_task_latents, correctness, strict=True):
        is_correct = bool(corr >= 0.5)
        if is_corrupt:
            out.append(-2.0 if is_correct else 0.0)
        else:
            out.append(2.0 if is_correct else 0.0)
    return out


def reward_stage4_scaled_correctness(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    label: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    corrupt_task_latents: Optional[List[bool]] = None,
    cot_len: Optional[List[int]] = None,
    **kwargs,
):
    """
    Stage 4 reward (latent-scaled):
    - If corrupted: reward = 0
    - If not corrupted:
        - correct: +4 / N
        - wrong:   -4 / N
      where N = CoT string length (`cot_len`, clamped to >= 1).
    """
    gt_list = labels if labels is not None else label
    if gt_list is None:
        return [0.0 for _ in completions]

    if corrupt_task_latents is None:
        corrupt_task_latents = [False for _ in completions]
    if cot_len is None:
        cot_len = [1 for _ in completions]
    correctness = reward_answer_correctness_bench(
        prompts=prompts,
        completions=completions,
        completion_ids=completion_ids,
        labels=gt_list,
        task=kwargs.get("task"),
        subtask=kwargs.get("subtask"),
        meta=kwargs.get("meta"),
    )
    out: list[float] = []
    for is_corrupt, corr, n in zip(corrupt_task_latents, correctness, cot_len, strict=True):
        if is_corrupt:
            out.append(0.0)
            continue
        n_cot = max(int(n), 1)
        is_correct = bool(corr >= 0.5)
        out.append((4.0 / n_cot) if is_correct else (-4.0 / n_cot))
    return out


def reward_stage4_double_scaled_correctness(
    prompts: List[str],
    completions: List[str],
    completion_ids=None,
    label: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    corrupt_task_latents: Optional[List[bool]] = None,
    task_latent_count: Optional[List[int]] = None,
    cot_len: Optional[List[int]] = None,
    **kwargs,
):
    """
    Stage 4 reward (double-scaled):
    - If corrupted: reward = 0
    - If not corrupted:
        - correct: +100 * (T / C)
        - wrong:   -100 / (T * C)
      where:
        - T = number of task latent tokens (`task_latent_count`, clamped to >= 0 for the correct case, and >= 1 for
              the wrong case to avoid division by zero)
        - C = CoT string length (`cot_len`, clamped to >= 1)
    """
    gt_list = labels if labels is not None else label
    if gt_list is None:
        return [0.0 for _ in completions]

    if corrupt_task_latents is None:
        corrupt_task_latents = [False for _ in completions]
    if task_latent_count is None:
        task_latent_count = [0 for _ in completions]
    if cot_len is None:
        cot_len = [1 for _ in completions]
    correctness = reward_answer_correctness_bench(
        prompts=prompts,
        completions=completions,
        completion_ids=completion_ids,
        labels=gt_list,
        task=kwargs.get("task"),
        subtask=kwargs.get("subtask"),
        meta=kwargs.get("meta"),
        corrupt_task_latents=corrupt_task_latents,
    )

    out: list[float] = []
    for is_corrupt, corr, t, c in zip(corrupt_task_latents, correctness, task_latent_count, cot_len, strict=True):
        if is_corrupt:
            out.append(0.0)
            continue

        t_cnt = int(t)
        c_len = max(int(c), 1)

        is_correct = bool(corr >= 0.5)
        if is_correct:
            out.append(100.0 * (float(t_cnt) / float(c_len)))
        else:
            denom = float(max(t_cnt, 1)) * float(c_len)
            out.append(-100.0 / denom)
    return out
