# ============================
# Part 1. Dataset loading & preprocessing
# ============================

import json
import re
from collections import OrderedDict
import torch
import os
import glob
from datasets import load_dataset
from transformers import AutoTokenizer
from config import ModelConfig
# --- Protein embedding loader (HDF5 index) ---
try:
    from load_protein_embeddings import load_protein_h5_encoder
    _HAS_PROTEIN_LOADER = True
except Exception:
    load_protein_h5_encoder = None
    _HAS_PROTEIN_LOADER = False

import tempfile
import pathlib
import shutil

# --------------------------------
# Load tokenizer (Qwen decoder-only LM)
# --------------------------------
tokenizer = AutoTokenizer.from_pretrained(ModelConfig.DEFAULT_QWEN_GENERAL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# 🚨 Coconut 特殊标记
COCONUT_TOKENS = {
    "latent": "<latent>",
    "start_latent": "<start_latent>",
    "end_latent": "<end_latent>",
    "mol_start": "<mol_start>",
    "mol_end": "<mol_end>"
}
# 确保所有特殊标记都添加到词表
tokenizer.add_tokens(list(COCONUT_TOKENS.values()))

LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["latent"])
START_LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["start_latent"])
END_LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["end_latent"])

# 最大文本长度（prompt + answer），从配置中读取
MAX_LEN = ModelConfig.MAX_TEXT_LEN


# --------------------------------
# 1. 从原始数据中抽取关键信息
# --------------------------------
def extract_fields(example):
    """
    从原始 ChemCot 数据中提取：
    - query: 作为 prompt
    - input_smiles: 分子 SMILES（用于多模态分子编码器）
    - label: 作为 LLM 的监督答案

    label 优先级：
        gt > reference > struct_cot 中解析出的 output
    """
    # meta 字段是一个 JSON 字符串，需要先解析
    # meta_dict = json.loads(example["meta"])
    raw_meta = example.get("meta", None)

    if raw_meta is None:
        meta_dict = {}
    elif isinstance(raw_meta, dict):
        meta_dict = raw_meta
    else:
        try:
            meta_dict = json.loads(raw_meta)
        except Exception:
            meta_dict = {}

    # 2. 解析 struct_cot
    # 第一步：初次解析，处理可能的 JSON 转义
    # try:
    #     cot_content = json.loads(example["struct_cot"], object_pairs_hook=OrderedDict)
    # except json.JSONDecodeError as e:
    #     print(f"\n[CRITICAL DATA ERROR] JSON is malformed in example ID: {example.get('id')}")
    #     print(f"[ERROR DETAILS]: {e}")
    #     print(f"[RAW CONTENT]: {repr(example.get('struct_cot'))}")
        # raise  # 依然抛出错误，中断训练
    
    # 2. 解析 struct_cot — defensive, KEEPING original prints
    raw_struct = example.get("struct_cot", None)
    cot_dict = {}

    if raw_struct is None:
        # Missing struct_cot → treat as empty
        cot_dict = {}
    else:
        # First attempt: normal JSON parse
        try:
            cot_content = json.loads(raw_struct, object_pairs_hook=OrderedDict)
        except json.JSONDecodeError as e:
            print(f"\n[CRITICAL DATA ERROR] JSON is malformed in example ID: {example.get('id')}")
            print(f"[ERROR DETAILS]: {e}")
            print(f"[RAW CONTENT]: {repr(raw_struct)}")
            # Instead of crashing training, fall back to empty CoT
            cot_content = None
        except TypeError:
            # Happens if raw_struct is not a string (e.g., already a dict)
            cot_content = raw_struct

        # If parsed content is a string, it may be ```json fenced
        if isinstance(cot_content, str):
            cleaned = cot_content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:-3].strip()
            try:
                cot_dict = json.loads(cleaned, object_pairs_hook=OrderedDict)
            except json.JSONDecodeError as e:
                print(f"\n[CRITICAL DATA ERROR] Secondary JSON parsing failed for example ID: {example.get('id')}")
                print(f"[ERROR DETAILS]: {e}")
                print(f"[CLEANED CONTENT]: {repr(cleaned)}")
                cot_dict = {}
        elif isinstance(cot_content, dict):
            cot_dict = cot_content
        else:
            cot_dict = {}
        
        # 第二步：如果解析出来是带 Markdown 标签的字符串，则剥离标签并进行二次解析
        if isinstance(cot_content, str):
            cleaned = cot_content.strip()
            if cleaned.startswith("```json"):
                # 移除开头的 ```json 和结尾的 ```
                cleaned = cleaned[7:-3].strip()
            # 二次解析，如果格式不对这里会直接报错
            try:
                cot_dict = json.loads(cleaned, object_pairs_hook=OrderedDict)
            except json.JSONDecodeError as e:
                print(f"\n[CRITICAL DATA ERROR] Secondary JSON parsing failed for example ID: {example.get('id')}")
                print(f"[ERROR DETAILS]: {e}")
                print(f"[CLEANED CONTENT]: {repr(cleaned)}")
                raise
        else:
            cot_dict = cot_content
    
    # 3. 构造 CoT 步骤列表 (Coconut 专用)
    # 每个步骤是一个字符串，例如 "Step 1:\nSMILES: CCC"
    cot_steps = []
    if not isinstance(cot_dict, dict):
        cot_dict = {}
    for i, (k, v) in enumerate(cot_dict.items()):
        if k == "output": continue # output 另外处理
        cot_steps.append(f"Step {i+1}:\n{k}: {v}")
    
    # 为了兼容旧版 SFT，依然保留 cot_value 字符串
    cot_value = "\n\n".join(cot_steps)

    # 4. 提取 label 优先级
    if meta_dict.get("gt"):
        label_value = str(meta_dict["gt"])
    elif meta_dict.get("reference"):
        label_value = str(meta_dict["reference"])
    else:
        label_value = str(cot_dict.get("output", ""))

    # 提取 SMILES
    raw_val = meta_dict.get("molecule")
    if raw_val is None:
        raw_val = meta_dict.get("reactants")
    
    # 统一转为列表处理
    if isinstance(raw_val, str):
        val_list = [raw_val]
    elif isinstance(raw_val, list):
        val_list = raw_val
    else:
        val_list = []
        
    # 按 '.' 切分并处理末尾点的情况，同时保持顺序
    input_smiles = []
    for s in val_list:
        if isinstance(s, str):
            # split('.') 会把 "C.C." 变成 ["C", "C", ""]
            # 通过 if part 过滤掉空字符串，正好相当于去掉了末尾的点或连续的点
            for part in s.split('.'):
                if part:
                    input_smiles.append(part)

    # 处理 query 中的 SMILES 标记
    query = example.get("query")
    
    # --------------------------------
    # 替换特定的 JSON 格式要求为 <answer> 格式
    # --------------------------------
    if query:
        # 1. 删除无意义的说明句子
        junk_patterns = [
            r'Do not provide any additional information beyond the requested SMILES strings\.?',
            r'The answer should be a json format that includes the potential byproduct SMILES:?',
            r'The answer should be a json format that includes the major product SMILES:?',
        ]
        for pattern in junk_patterns:
            query = re.sub(pattern, "", query, flags=re.IGNORECASE)

        # 标记是否成功匹配并替换了任何 JSON 格式块
        matched_format = False

        # 2. 分开匹配不同的引导语和对应的 Key 块
        # 处理 "Your response must be" 类型
        your_response_keys = {
            "Final Target Molecule": "SMILES",
            "Output Scaffold": "SMILES",
            "count": "Your Answer Number",
            "output": "Yes / No"
        }
        for key, placeholder in your_response_keys.items():
            pattern = rf'Your response must be[^{{]*?\{{[^}}]*?"{key}"[^}}]*?\}}'
            # 使用 subn 获取替换次数 n
            query, n = re.subn(pattern, f'Your final answer must be formatted as <answer> {placeholder} </answer>', query, flags=re.DOTALL)
            if n > 0:
                matched_format = True

        # 处理 "Answer:" 类型
        answer_keys = {
            "By Product": "SMILES",
            "Major Product": "SMILES"
        }
        for key, placeholder in answer_keys.items():
            pattern = rf'Answer:[^{{]*?\{{[^}}]*?"{key}"[^}}]*?\}}'
            query, n = re.subn(pattern, f'Your final answer must be formatted as <answer> {placeholder} </answer>', query, flags=re.DOTALL)
            if n > 0:
                matched_format = True
        
        # 如果没有任何特定的 JSON 块被匹配上，追加默认格式指令
        if not matched_format:
            query = query.rstrip() + "\nYour final answer must be formatted as <answer> Your Answer </answer>"
        
        query = query.strip()

    if query and input_smiles:
        # 1. 按长度从长到短排序，防止短 SMILES (如 C) 误匹配长 SMILES (如 CC) 的一部分
        indexed_smiles = sorted(enumerate(input_smiles), key=lambda x: len(x[1]), reverse=True)
        
        # 边界检查字符集：防止误伤单词（Cat）或长链内部（C1...）
        smiles_chars = r'a-zA-Z0-9\[\]\(\)\=#@+\-\/\\%'
        
        for i, s in indexed_smiles:
            # 使用正则进行边界检查，确保匹配的是独立的 SMILES 实体
            pattern = rf'(?<![{smiles_chars}]){re.escape(s)}(?![{smiles_chars}])'
            if re.search(pattern, query):
                # 使用 lambda 替换，避免 re.sub 对 SMILES 中反斜杠 (\) 的错误转义
                replacement = f"{s} (the {i+1}-th SMILES)"
                query = re.sub(pattern, lambda m: replacement, query)
            else:
                # print(f"SMILES not found in query: {s}")
                pass

    return {
        # LLM 输入的文本 prompt
        "query": query,
        # 分子 SMILES 列表
        "input_smiles": input_smiles,
        # LLM 的监督答案
        "label": f"<answer> {label_value} </answer>",
        # Benchmark routing (used by GRPO rewards)
        "task": example.get("task"),
        "subtask": example.get("subtask"),
        "meta": example.get("meta"),
        # 结构化思维链 (CoT)
        "cot": cot_value,
        # CoT 字符长度（用于动态分配 latent 数量）
        "cot_len": len(cot_value) if cot_value is not None else 0,
        # 分步思维链 (Coconut 专用)
        "cot_steps": cot_steps,
    }

def attach_protein_keys(example, available_keys=None, map_random=False, random_count=1):
    """
    Adds: example["protein_keys"] (list[str])

    Priority:
    1) If meta contains 'protein_keys' (list) or 'protein_key' (str) -> use them.
    2) Else if map_random=True -> sample random keys from available_keys.
    3) Else -> [] (no proteins attached).
    """
    meta = {}
    try:
        meta = json.loads(example.get("meta", "{}"))
    except Exception:
        meta = {}

    if isinstance(meta.get("protein_keys"), list):
        return {"protein_keys": meta["protein_keys"]}
    if isinstance(meta.get("protein_key"), str):
        return {"protein_keys": [meta["protein_key"]]}

    if map_random and available_keys:
        import random
        picked = random.sample(available_keys, min(int(random_count), len(available_keys)))
        return {"protein_keys": picked}

    return {"protein_keys": []}


# --------------------------------
# 2.5 构造 Coconut 训练样本
# --------------------------------
def coconut_tokenize(
    example, 
    scheduled_stage=0, 
    c_thought=2, 
    max_len=ModelConfig.MAX_TEXT_LEN
):
    """
    Coconut 训练的核心数据处理：
    将前 scheduled_stage 个步骤替换为 (scheduled_stage * c_thought) 个 <latent> tokens。
    """
    prompt = example["query"]
    steps = example["cot_steps"]
    label = example["label"]
    
    # 确定要替换的步数（不能超过总步数）
    n_skip_steps = min(len(steps), scheduled_stage)
    n_latent_tokens = n_skip_steps * c_thought
    
    # 1. Prompt 部分 Tokenize
    prompt_ids = tokenizer.encode(f"{prompt}\n\n", add_special_tokens=False)
    
    # 2. Latent 部分拼接
    # 格式：<start_latent> + <latent> * N + <end_latent>
    latent_ids = [START_LATENT_ID] + [LATENT_ID] * n_latent_tokens + [END_LATENT_ID]
    
    # 3. 剩余文本步骤 Tokenize
    remaining_steps_text = "\n\n".join(steps[n_skip_steps:])
    if remaining_steps_text:
        remaining_steps_text += "\n\n"
    
    response_text = f"{remaining_steps_text}{label}{tokenizer.eos_token}"
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    
    # 4. 全局拼接
    input_ids = (prompt_ids + latent_ids + response_ids)[:max_len]
    attention_mask = [1] * len(input_ids)
    
    # 5. 构造 labels
    # Prompt 和 Latent 部分都需要 mask 掉 (-100)
    # 只有剩余的文本步骤和最后的答案计算 Loss
    labels = input_ids.copy()
    mask_len = min(len(prompt_ids) + len(latent_ids), max_len)
    labels[:mask_len] = [-100] * mask_len
    
    # 额外：为了方便 Coconut 模型的迭代 forward，我们需要记录这些 latent token 的位置索引
    # 虽然 Trainer 会做 padding，但我们在 forward 内部会重新寻找
    
    # return {
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "labels": labels,
    #     "smiles": example["input_smiles"],
    # }
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "smiles": example["input_smiles"],
        "protein_keys": example.get("protein_keys", []),  # NEW
    }


# --------------------------------
# 2. 构造 Causal LM 的训练样本 (旧版 SFT 兼容)
# --------------------------------
def llm_tokenize(example, include_cot=True, max_len=ModelConfig.MAX_TEXT_LEN):
    """
    构造 Causal Language Model 的训练格式：
    使用分别 Tokenize 再拼接的方法，确保 Label 对齐绝对精确。
    """

    prompt = example["query"]
    cot = example.get("cot", "")
    label = example["label"]

    # 根据参数决定是否包含 CoT
    if include_cot and cot:
        response = f"{cot}\n\n{label}"
    else:
        response = label
    
    # 1. Tokenize Prompt 部分 (包括分隔符)
    prompt_enc = tokenizer(
        f"{prompt}\n\n",
        truncation=True,
        padding=False,
        max_length=max_len,
        add_special_tokens=False # 避免重复添加 bos_token
    )
    
    # 2. Tokenize Response 部分 (包括结束符)
    response_enc = tokenizer(
        f"{response}{tokenizer.eos_token}",
        truncation=True,
        padding=False,
        max_length=max_len,
        add_special_tokens=False
    )

    prompt_ids = prompt_enc["input_ids"]
    response_ids = response_enc["input_ids"]

    # 拼接并截断到 max_len
    input_ids = (prompt_ids + response_ids)[:max_len]
    attention_mask = (prompt_enc["attention_mask"] + response_enc["attention_mask"])[:max_len]

    # -------- 构造 labels --------
    # 初始 labels 与 input_ids 相同
    labels = input_ids.copy()

    # 精确计算 prompt 长度（考虑截断情况）
    actual_prompt_len = min(len(prompt_ids), max_len)

    # 将 prompt 部分的 label mask 掉
    labels[:actual_prompt_len] = [-100] * actual_prompt_len

    # return {
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "labels": labels,
    #     "smiles": example["input_smiles"],  # Fixed: changed from example["smiles"] to example["input_smiles"]
    # }
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "smiles": example["input_smiles"],
        "protein_keys": example.get("protein_keys", []),  # NEW
    }


# --------------------------------
# 3. 数据集加载与整体处理流程
# --------------------------------
# def load_data(
#     path, 
#     include_cot=True, 
#     max_len=ModelConfig.MAX_TEXT_LEN,
#     is_coconut=False,
#     scheduled_stage=0,
#     c_thought=2
# ):
def load_data(
    path,
    include_cot=True,
    max_len=ModelConfig.MAX_TEXT_LEN,
    is_coconut=False,
    scheduled_stage=0,
    c_thought=2,
    include_proteins=False,                 # NEW
    protein_embeddings_folder=None,         # NEW: path to embeddings_proteins/
    map_random_proteins=False,              # NEW
    random_protein_count=1                  # NEW
):
    """
    完整的数据加载流程：
    1. 加载原始 ChemCot 数据集
    2. 提取 query / smiles / label / steps
    3. 将文本转为 LLM 可训练的 token 格式 (支持 SFT 或 Coconut)
    """

    # 扫描所有 JSON 文件并排除 rxn/rcr.json
    all_json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)
    data_files = [f for f in all_json_files if not f.endswith("rcr.json")]
    
    # print(f"Loading {len(data_files)} JSON files...")
    
    # 加载过滤后的数据文件
    # ds = load_dataset("json", data_files=data_files)["train"]
    # --- Robust load: some files have nested `annots` structs that break Arrow schema ---
    try:
        ds = load_dataset("json", data_files=data_files)["train"]
    except Exception as orig_e:
        print("load_dataset(json) failed; normalizing JSON -> JSONL and reloading.")
        print("Original error:", repr(orig_e))

        tmp_dir = tempfile.mkdtemp(prefix="normalized_json_")
        tmp_paths = []
        promote_keys = ["num_annots", "num_tokens_in_input", "protein_accession", "seq_len", "split", "task"]

        try:
            for src in sorted(data_files):
                src_path = pathlib.Path(src)
                dst_path = pathlib.Path(tmp_dir) / (src_path.stem + ".jsonl")
                tmp_paths.append(str(dst_path))

                with src_path.open("r", encoding="utf-8") as fh_in:
                    try:
                        obj = json.load(fh_in)
                    except Exception as e:
                        print(f"Warning: failed to parse JSON {src}: {e}")
                        continue

                # file may be list/dict/single
                if isinstance(obj, list):
                    records = obj
                elif isinstance(obj, dict):
                    lists = [v for v in obj.values() if isinstance(v, list)]
                    if lists:
                        records = []
                        for lst in lists:
                            records.extend(lst)
                    else:
                        records = [obj]
                else:
                    records = [obj]

                with dst_path.open("w", encoding="utf-8") as fh_out:
                    for rec in records:
                        if not isinstance(rec, dict):
                            continue

                        # Normalize annots struct -> promote fields + serialize annots as string
                        ann = rec.get("annots")
                        if isinstance(ann, dict):
                            for k in promote_keys:
                                if k in ann and k not in rec:
                                    rec[k] = ann[k]
                            rec["annots"] = json.dumps(ann, ensure_ascii=False)

                        # Ensure meta is JSON string (your extract_fields expects string)
                        if "meta" in rec and isinstance(rec["meta"], dict):
                            rec["meta"] = json.dumps(rec["meta"], ensure_ascii=False)

                        # Any other dict fields -> serialize to string to avoid Arrow struct types
                        for k, v in list(rec.items()):
                            if isinstance(v, dict) and k not in ("annots", "meta"):
                                rec[k] = json.dumps(v, ensure_ascii=False)

                        fh_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

            ds = load_dataset("json", data_files=tmp_paths, split="train")
        finally:
            try:
                shutil.rmtree(tmp_dir)
            except Exception as cleanup_e:
                print(f"Warning: temp cleanup failed: {cleanup_e}")

    # 过滤已知损坏的数据 ID
    bad_ids = [
        "f7e567a6-47de-4c77-8c1f-9049689322e8",
        "bedfe3e8-ab07-4b8e-b872-ae281e5f55af",
        "9cb0a77d-6203-4686-9c8b-45fd3fc770f2"
    ]
    # ds = ds.filter(lambda x: x["id"] not in bad_ids)
    # Ensure we have an 'id' column (some JSONs lack it). If missing, add synthetic ids.
    if "id" not in ds.column_names:
        # create stable synthetic string ids based on dataset index
        ds = ds.map(lambda example, idx: {"id": str(idx)}, with_indices=True)

    # Now safely filter out known-bad ids (use .get to be extra-safe)
    ds = ds.filter(lambda x: x.get("id") not in bad_ids)

    # --------------------------------
    # Step 1: 提取结构化字段
    # --------------------------------
    dataset = ds.map(
        extract_fields,
        batched=False,
        remove_columns=ds.column_names
    )

    # ---- Attach protein_keys so the model can load ESM embeddings from .h5 ----
    if include_proteins:
        if not _HAS_PROTEIN_LOADER:
            raise RuntimeError("include_proteins=True but load_protein_h5_encoder could not be imported.")
        if protein_embeddings_folder is None:
            raise ValueError("include_proteins=True requires protein_embeddings_folder (e.g. 'embeddings_proteins').")

                # Build list of valid keys from HDF5 index (lazy/open depending on loader version)
        try:
            # Try newer signature (accepts lazy_open)
            protein_encoder = load_protein_h5_encoder(
                folder=protein_embeddings_folder,
                device="cpu",
                pool="none",
                lazy_open=True,
            )
        except TypeError:
            # Fallback for older loader implementations that don't accept lazy_open
            # Try without lazy_open. If the loader has a different signature, this will raise and you'll see the error.
            print("load_protein_h5_encoder() does not accept 'lazy_open'; retrying without it.")
            protein_encoder = load_protein_h5_encoder(
                folder=protein_embeddings_folder,
                device="cpu",
                pool="none",
            )

        # Build available keys list (works for lazy or eager loader)
        try:
            available_keys = list(protein_encoder._index.keys())
        except Exception:
            # Some loaders expose a different API; try a couple common alternatives
            if hasattr(protein_encoder, "index"):
                available_keys = list(protein_encoder.index.keys())
            elif hasattr(protein_encoder, "_files"):
                # fallback: if loader keeps list of files
                available_keys = list(protein_encoder._files)
            else:
                # give a helpful error
                raise RuntimeError(
                    "Could not enumerate keys from the protein encoder object. "
                    "Inspect the encoder's attributes (e.g. _index, index, _files) to find available keys."
                )

        dataset = dataset.map(
            lambda ex: attach_protein_keys(
                ex,
                available_keys=available_keys,
                map_random=map_random_proteins,
                random_count=random_protein_count,
            ),
            batched=False,
        )
    else:
        # Always provide the field so downstream collators/models can rely on it
        dataset = dataset.map(lambda ex: {"protein_keys": []}, batched=False)

    # --------------------------------
    # Step 2: 构造训练样本
    # --------------------------------
    if is_coconut:
        # Coconut 模式
        dataset = dataset.map(
            coconut_tokenize,
            batched=False,
            fn_kwargs={
                "scheduled_stage": scheduled_stage, 
                "c_thought": c_thought, 
                "max_len": max_len
            },
            remove_columns=["query", "input_smiles", "label", "cot", "cot_steps", "task", "subtask", "meta"]
        )
    else:
        # 标准 SFT 模式
        dataset = dataset.map(
            llm_tokenize,
            batched=False,
                fn_kwargs={"include_cot": include_cot, "max_len": max_len},
                remove_columns=["query", "input_smiles", "label", "cot", "cot_steps", "task", "subtask", "meta"]
        )

    return dataset


# --------------------------------
# 3b. GRPO prompt-only dataset (for RL)
# --------------------------------
def load_grpo_data(path):
    """
    Load a prompt-only dataset for GRPO-style RL training.

    Returns a HuggingFace `Dataset` with (at least):
    - `prompt`: str
    - `input_smiles`: list[str]
    - `label`: str (ground-truth answer wrapped as `<answer> ... </answer>`, used for reward shaping)

    Tokenization/collation should be handled by the GRPO trainer's collate function.
    """
    # 扫描所有 JSON 文件并排除 rxn/rcr.json
    all_json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)
    data_files = [f for f in all_json_files if not f.endswith("rcr.json")]
    from datasets import load_dataset
    ds = load_dataset("json", data_files=data_files)["train"]

    # 过滤已知损坏的数据 ID
    bad_ids = [
        "f7e567a6-47de-4c77-8c1f-9049689322e8",
        "bedfe3e8-ab07-4b8e-b872-ae281e5f55af",
        "9cb0a77d-6203-4686-9c8b-45fd3fc770f2",
    ]
    ds = ds.filter(lambda x: x["id"] not in bad_ids)

    dataset = ds.map(
        extract_fields,
        batched=False,
        remove_columns=ds.column_names,
    )

    # Keep only what GRPO needs
    dataset = dataset.rename_column("query", "prompt")
    dataset = dataset.remove_columns(
        [
            c
            for c in dataset.column_names
            if c not in ("prompt", "input_smiles", "label", "task", "subtask", "meta", "cot_len")
        ]
    )
    return dataset


# --------------------------------
# 4. 运行示例与测试
# --------------------------------
if __name__ == "__main__":
    # 测试路径
    # DATA_ROOT = "/mnt/afs/L202500070/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot"
    
    # # 获取所有 JSON 文件
    # all_json_files = glob.glob(os.path.join(DATA_ROOT, "**/*.json"), recursive=True)
    # test_files = [f for f in all_json_files if not f.endswith("rcr.json")]
    
    # print(f"\n{'='*20} Testing Query Replacement {'='*20}")
    
    # for f_path in sorted(test_files):
    #     rel_path = os.path.relpath(f_path, DATA_ROOT)
    #     try:
    #         with open(f_path, 'r', encoding='utf-8') as f:
    #             data = json.load(f)
            
    #         if not data:
    #             continue
            
    #         # 如果是 fs_by_product.json，测试前 15 条，否则测试第 1 条
    #         num_to_test = 15 if "fs_by_product.json" in rel_path else 1
            
    #         print(f"\n{'#'*10} [FILE]: {rel_path} (Testing {num_to_test} samples) {'#'*10}")
            
    #         for i in range(min(len(data), num_to_test)):
    #             example = data[i]
    #             processed = extract_fields(example)
                
    #             print(f"\n--- Sample {i+1} ---")
    #             for key, val in processed.items():
    #                 print(f"[{key.upper()}]:\n{val}\n")
            
    #         print(f"\n{'='*60}")
            
    #     except Exception as e:
    #         print(f"Error processing {rel_path}: {e}")

    # 也可以保留原有的全量加载测试（可选）
    # dataset = load_data(DATA_ROOT)
    # print(f"\nTotal samples loaded: {len(dataset)}")

    # CHECK THE DATALOADER FOR PROTEINS WORKS
    DATA_ROOT = ModelConfig.CURRENT_DIR
    dataset = load_data(
        DATA_ROOT,
        include_cot=True,
        is_coconut=False,
        include_proteins=True,
        protein_embeddings_folder="embeddings_proteins",  # folder containing *_esm2_650m.h5
        map_random_proteins=True,
        random_protein_count=1
    )
    print(f"\nTotal samples loaded: {len(dataset)}")