# dataloader_proteins_only.py
"""
Protein-only dataset loader and preprocessing.

- Loads only selected protein-oriented JSON files from Mol-Instructions.
- Extracts protein sequences (if present) from metadata / input / instruction / output.
- Builds train/validation/test DatasetDict from metadata["split"].
- Tokenizes using the Qwen tokenizer and supports Coconut or standard SFT token formats.
"""

import json
import re
from collections import OrderedDict
import os
import glob
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from config_proteins import ModelConfig
import torch

# -------------------------
# Tokenizer / special tokens
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(ModelConfig.DEFAULT_QWEN_PATH)
tokenizer.pad_token = tokenizer.eos_token

# Coconut special tokens (no molecule tokens here)
COCONUT_TOKENS = {
    "latent": "<latent>",
    "start_latent": "<start_latent>",
    "end_latent": "<end_latent>",
}
tokenizer.add_tokens(list(COCONUT_TOKENS.values()))

LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["latent"])
START_LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["start_latent"])
END_LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["end_latent"])

MAX_LEN = ModelConfig.MAX_TEXT_LEN

# =========================
# Protein extraction utils
# =========================
# AA regex: 10+ canonical amino acids (simple heuristic)
AA_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{10,}", re.IGNORECASE)
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def is_valid_protein_sequence(seq: str, min_len: int = 20, max_len: int = 5000) -> bool:
    """Return True if `seq` looks like a canonical protein sequence."""
    if not isinstance(seq, str):
        return False
    s = seq.replace(" ", "").replace("\n", "").upper()
    if not (min_len <= len(s) <= max_len):
        return False
    return all(ch in CANONICAL_AA for ch in s)


def extract_proteins_from_meta(meta) -> list:
    """
    Inspect metadata structure (dict or json-string) and return list of protein sequences
    found under common keys or embedded in text fields.
    """
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            # leave as-is if not JSON
            meta = {}

    if not isinstance(meta, dict):
        meta = {}

    candidates = []
    # common metadata keys that *might* contain sequences or lists
    for key in ("protein", "protein_sequence", "sequence", "target_sequence", "seq", "input_sequence"):
        v = meta.get(key)
        if isinstance(v, str):
            candidates.append(v)
        elif isinstance(v, list):
            candidates.extend([x for x in v if isinstance(x, str)])

    # sometimes sequences are stored under "molecule" (rare for proteins, but keep)
    mol = meta.get("molecule")
    if isinstance(mol, str):
        candidates.append(mol)
    elif isinstance(mol, list):
        candidates.extend([x for x in mol if isinstance(x, str)])

    # Also look inside 'annots' or 'ann' if it's long text
    ann = meta.get("annots") or meta.get("annotation") or meta.get("notes")
    if isinstance(ann, str):
        candidates.append(ann)

    proteins = []
    for s in candidates:
        if not isinstance(s, str):
            continue
        clean = s.replace("\n", "").replace(" ", "").upper()
        if is_valid_protein_sequence(clean):
            proteins.append(clean)
        else:
            m = AA_RE.search(s)
            if m:
                proteins.append(m.group(0).replace("\n", "").upper())

    # dedupe preserving order
    seen = set()
    out = []
    for p in proteins:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# =========================
# Field extraction
# =========================
def extract_fields(example: dict) -> dict:
    """
    Extract structured fields from a raw example.
    Returned dict contains:
      - query: textual instruction/prompt (from 'instruction' or 'input')
      - input_proteins: list[str] of sequences found
      - label: "<answer> ... </answer>" derived from metadata or output
      - cot: chain-of-thought string (if parseable)
      - cot_len, cot_steps: helper fields for Coconut
      - metadata: original metadata (kept so we can split by 'split')
    """
    # Many files use "metadata" (dict) rather than "meta"; handle both.
    raw_meta = example.get("metadata") or example.get("meta") or {}
    # Normalize metadata as dict or JSON string; keep original in 'metadata' field
    meta_dict = raw_meta if isinstance(raw_meta, dict) else (json.loads(raw_meta) if isinstance(raw_meta, str) else {})

    # Parse any structured CoT if present (struct_cot key may not exist in these files)
    struct_cot_raw = example.get("struct_cot", "") or example.get("struct_cot", "")
    cot_steps = []
    cot_value = ""
    cot_dict = None
    if struct_cot_raw:
        try:
            cot_content = json.loads(struct_cot_raw, object_pairs_hook=OrderedDict)
        except Exception:
            cot_content = struct_cot_raw

        if isinstance(cot_content, str):
            cleaned = cot_content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:-3].strip()
            try:
                cot_dict = json.loads(cleaned, object_pairs_hook=OrderedDict)
            except Exception:
                cot_dict = None
        else:
            cot_dict = cot_content

    if isinstance(cot_dict, dict):
        for i, (k, v) in enumerate(cot_dict.items()):
            if k == "output":
                continue
            cot_steps.append(f"Step {i+1}:\n{k}: {v}")
        cot_value = "\n\n".join(cot_steps)
    else:
        # fallback: if struct_cot_raw is plain text, split heuristically
        if isinstance(struct_cot_raw, str) and struct_cot_raw.strip():
            parts = [p.strip() for p in re.split(r"\n{2,}", struct_cot_raw) if p.strip()]
            for i, p in enumerate(parts):
                cot_steps.append(f"Step {i+1}:\n{p}")
            cot_value = "\n\n".join(cot_steps)

    # Determine label: check metadata fields 'gt' or 'reference' or fallback to example['output']
    if meta_dict.get("gt"):
        label_value = str(meta_dict["gt"])
    elif meta_dict.get("reference"):
        label_value = str(meta_dict["reference"])
    elif isinstance(cot_dict, dict) and "output" in cot_dict:
        label_value = str(cot_dict["output"])
    else:
        label_value = str(example.get("output", "") or "")

    # Try to extract protein sequences from metadata first
    input_proteins = extract_proteins_from_meta(meta_dict)

    # Fallback: search in 'input', 'instruction', 'output' fields for AA substrings
    if not input_proteins:
        text_sources = [
            example.get("input", "") or example.get("query", "") or "",
            example.get("instruction", "") or "",
            example.get("output", "") or "",
            cot_value
        ]
        for txt in text_sources:
            if not isinstance(txt, str):
                continue
            for match in AA_RE.findall(txt):
                cand = match.replace("\n", "").replace(" ", "").upper()
                if is_valid_protein_sequence(cand):
                    input_proteins.append(cand)

    # dedupe preserving order
    seen = set()
    proteins_final = []
    for p in input_proteins:
        if p not in seen:
            seen.add(p)
            proteins_final.append(p)
    input_proteins = proteins_final

    # Choose a query field (prefer 'instruction', then 'input')
    query = example.get("instruction") or example.get("input") or example.get("query") or ""

    return {
        "query": query,
        "input_proteins": input_proteins,
        "label": f"<answer> {label_value} </answer>",
        "cot": cot_value,
        "cot_len": len(cot_value) if cot_value else 0,
        "cot_steps": cot_steps,
        "metadata": meta_dict,  # keep metadata dict for splitting and provenance
    }


# =========================
# Coconut tokenization
# =========================
def coconut_tokenize(example, scheduled_stage=0, c_thought=2, max_len=MAX_LEN):
    """
    Tokenize for Coconut-style training (latent tokens replace early CoT steps).
    Returns tokenized inputs and preserves `input_proteins` field.
    """
    prompt = example.get("query", "")
    steps = example.get("cot_steps", [])
    label = example.get("label", "")

    n_skip = min(len(steps), scheduled_stage)
    n_latent = n_skip * c_thought

    prompt_ids = tokenizer.encode(f"{prompt}\n\n", add_special_tokens=False)
    latent_ids = [START_LATENT_ID] + [LATENT_ID] * n_latent + [END_LATENT_ID]

    remaining = "\n\n".join(steps[n_skip:])
    if remaining:
        remaining += "\n\n"

    response_ids = tokenizer.encode(f"{remaining}{label}{tokenizer.eos_token}", add_special_tokens=False)

    input_ids = (prompt_ids + latent_ids + response_ids)[:max_len]
    attention_mask = [1] * len(input_ids)

    labels = input_ids.copy()
    mask_len = min(len(prompt_ids) + len(latent_ids), max_len)
    labels[:mask_len] = [-100] * mask_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_proteins": example.get("input_proteins", []),
        "metadata": example.get("metadata", {}),
    }


# =========================
# Standard SFT tokenization
# =========================
def llm_tokenize(example, include_cot=True, max_len=MAX_LEN):
    """Standard causal-LM tokenization; returns input_proteins and metadata too."""
    prompt = example.get("query", "")
    cot = example.get("cot", "")
    label = example.get("label", "")

    response = f"{cot}\n\n{label}" if include_cot and cot else label

    prompt_enc = tokenizer(f"{prompt}\n\n", truncation=True, max_length=max_len, add_special_tokens=False)
    response_enc = tokenizer(f"{response}{tokenizer.eos_token}", truncation=True, max_length=max_len, add_special_tokens=False)

    prompt_ids = prompt_enc["input_ids"]
    response_ids = response_enc["input_ids"]

    input_ids = (prompt_ids + response_ids)[:max_len]
    attention_mask = (prompt_enc["attention_mask"] + response_enc["attention_mask"])[:max_len]

    labels = input_ids.copy()
    actual_prompt_len = min(len(prompt_ids), max_len)
    labels[:actual_prompt_len] = [-100] * actual_prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_proteins": example.get("input_proteins", []),
        "metadata": example.get("metadata", {}),
    }


# =========================
# Load pipeline
# =========================
FILES_TO_PROCESS = [
    "protein_function",
    "catalytic_activity",
    "domain_motif",
]  # basenames to include (edit if needed)


def load_data(
    path,
    include_cot=True,
    is_coconut=False,
    scheduled_stage=0,
    c_thought=2,
):
    """
    Load only protein-oriented files under `path`, extract protein fields, split by metadata['split'],
    tokenize each split and return a DatasetDict with train/validation/test.
    """
    # Find all JSON files under path that belong to Protein-oriented_Instructions
    all_json_files = sorted(glob.glob(os.path.join(path, "**/*.json"), recursive=True))
    # filter files to only the protein-oriented directory or those whose basename matches FILES_TO_PROCESS
    data_files = []
    for f in all_json_files:
        if "Protein-oriented_Instructions" in f.replace("\\", "/"):
            # check if basename contains any target base name
            base = os.path.splitext(os.path.basename(f))[0]
            for name in FILES_TO_PROCESS:
                # accept either exact basename or when base startswith the target (robust)
                if name == base or base.startswith(name):
                    data_files.append(f)
                    break

    if not data_files:
        raise RuntimeError(f"No protein JSON files found under {path}. Checked {len(all_json_files)} JSONs.")

    # Load dataset from the selected files
    ds = load_dataset("json", data_files=data_files, split="train")

    # Remove known-bad IDs if present
    bad_ids = {
        "f7e567a6-47de-4c77-8c1f-9049689322e8",
        "bedfe3e8-ab07-4b8e-b872-ae281e5f55af",
        "9cb0a77d-6203-4686-9c8b-45fd3fc770f2",
    }
    # Only filter if ds has 'id' column
    if "id" in ds.column_names:
        ds = ds.filter(lambda x: x.get("id") not in bad_ids)

    # Extract fields (keeps metadata)
    dataset = ds.map(extract_fields, batched=False, remove_columns=ds.column_names)

    # --------------------------
    # Split dataset by metadata['split']
    # --------------------------
    def _get_split_from_example(ex):
        md = ex.get("metadata") or {}
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except Exception:
                md = {}
        if not isinstance(md, dict):
            md = {}
        sp = md.get("split") or md.get("set") or md.get("dataset_split")
        return sp.lower() if isinstance(sp, str) else None

    train_ds = dataset.filter(lambda ex: _get_split_from_example(ex) == "train")
    val_ds = dataset.filter(lambda ex: _get_split_from_example(ex) in ("val", "validation", "dev"))
    test_ds = dataset.filter(lambda ex: _get_split_from_example(ex) == "test")

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    # --------------------------
    # Tokenize each split separately (keeps metadata until after split)
    # --------------------------
    if is_coconut:
        dataset_dict["train"] = dataset_dict["train"].map(
            coconut_tokenize,
            batched=False,
            fn_kwargs={"scheduled_stage": scheduled_stage, "c_thought": c_thought, "max_len": MAX_LEN},
            remove_columns=["query", "input_proteins", "label", "cot", "cot_steps", "metadata"],
        )
        dataset_dict["validation"] = dataset_dict["validation"].map(
            coconut_tokenize,
            batched=False,
            fn_kwargs={"scheduled_stage": scheduled_stage, "c_thought": c_thought, "max_len": MAX_LEN},
            remove_columns=["query", "input_proteins", "label", "cot", "cot_steps", "metadata"],
        )
        dataset_dict["test"] = dataset_dict["test"].map(
            coconut_tokenize,
            batched=False,
            fn_kwargs={"scheduled_stage": scheduled_stage, "c_thought": c_thought, "max_len": MAX_LEN},
            remove_columns=["query", "input_proteins", "label", "cot", "cot_steps", "metadata"],
        )
    else:
        # standard SFT tokenization
        dataset_dict["train"] = dataset_dict["train"].map(
            llm_tokenize,
            batched=False,
            fn_kwargs={"include_cot": include_cot, "max_len": MAX_LEN},
            remove_columns=["query", "input_proteins", "label", "cot", "cot_steps", "metadata"],
        )
        dataset_dict["validation"] = dataset_dict["validation"].map(
            llm_tokenize,
            batched=False,
            fn_kwargs={"include_cot": include_cot, "max_len": MAX_LEN},
            remove_columns=["query", "input_proteins", "label", "cot", "cot_steps", "metadata"],
        )
        dataset_dict["test"] = dataset_dict["test"].map(
            llm_tokenize,
            batched=False,
            fn_kwargs={"include_cot": include_cot, "max_len": MAX_LEN},
            remove_columns=["query", "input_proteins", "label", "cot", "cot_steps", "metadata"],
        )

    return dataset_dict


# =========================
# Quick test / demo when run as script
# =========================
if __name__ == "__main__":
    DATA_ROOT = "Mol-Instructions/data/proteins/Protein-oriented_Instructions"  # adjust as needed
    ds_dict = load_data(DATA_ROOT, include_cot=False, is_coconut=False)
    print(ds_dict)
    for split in ds_dict:
        print(f"{split} size: {len(ds_dict[split])}")
        # show a single example
        if len(ds_dict[split]) > 0:
            ex = ds_dict[split][0]
            print({k: ex.get(k) for k in ("input_ids", "attention_mask", "labels", "input_proteins") if k in ex})

    # 1) Check the special tokens exist and show their token ids
    print("Coconut tokens and ids:")
    for name, tok in COCONUT_TOKENS.items():
        print(name, tok, "-> id:", tokenizer.convert_tokens_to_ids(tok))

    # 2) Confirm none of the token ids are the unknown token id (basic sanity)
    unk = tokenizer.unk_token_id
    print("unk_token_id:", unk)
    print("Any token equal to unk?:", any(tokenizer.convert_tokens_to_ids(t) == unk for t in COCONUT_TOKENS.values()))

    # 3) Load dataset in SFT (non-coconut) mode and show splits and first tokenized example
    DATA_ROOT = "Mol-Instructions/data/proteins/Protein-oriented_Instructions"  # adjust
    ds_dict = load_data(DATA_ROOT, include_cot=True, is_coconut=False)
    print(ds_dict)
    for s in ("train","validation","test"):
        print(s, "size:", len(ds_dict[s]))

    # show first tokenized example from train if exists
    if len(ds_dict["train"])>0:
        ex = ds_dict["train"][0]
        print("train[0] keys:", list(ex.keys()))
        # if tokenized, will have input_ids etc
        for k in ("input_ids","attention_mask","labels","input_proteins"):
            print(k, "present:", k in ex)

    # 4) Load dataset in Coconut mode to test latent token insertion
    ds_dict_c = load_data(DATA_ROOT, include_cot=False, is_coconut=True, scheduled_stage=1, c_thought=2)
    print("Coconut tokenized train size:", len(ds_dict_c["train"]))
    if len(ds_dict_c["train"])>0:
        ex = ds_dict_c["train"][0]
        print({k: (ex[k][:20] if k=="input_ids" else ex.get(k)) for k in ex.keys() if k in ("input_ids","labels","input_proteins")})