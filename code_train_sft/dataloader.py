# ============================
# Part 1. Dataset loading & preprocessing (with eval mode)
# ============================

import json
import re
from collections import OrderedDict
import torch
import os
import glob
import random
from datasets import load_dataset
from transformers import AutoTokenizer
from config import ModelConfig
import ast

# Load tokenizer (Qwen decoder-only LM)
# --------------------------------
tokenizer = AutoTokenizer.from_pretrained(ModelConfig.DEFAULT_QWEN_PATH)
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


#
# --------------------------------
# Load tokenizer (Qwen decoder-only LM) - Lazy loading
# --------------------------------
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(ModelConfig.DEFAULT_QWEN_PATH, use_fast=False)
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

    return tokenizer

# Global variable to store the tokenizer once loaded
_tokenizer_instance = None

def get_tokenizer_with_cache():
    global _tokenizer_instance
    if _tokenizer_instance is None:
        _tokenizer_instance = get_tokenizer()
    return _tokenizer_instance

def get_tokenizer_constants():
    tokenizer = get_tokenizer_with_cache()
    COCONUT_TOKENS = {
        "latent": "<latent>",
        "start_latent": "<start_latent>",
        "end_latent": "<end_latent>",
        "mol_start": "<mol_start>",
        "mol_end": "<mol_end>"
    }
    LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["latent"])
    START_LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["start_latent"])
    END_LATENT_ID = tokenizer.convert_tokens_to_ids(COCONUT_TOKENS["end_latent"])
    return tokenizer, COCONUT_TOKENS, LATENT_ID, START_LATENT_ID, END_LATENT_ID

# 最大文本长度（prompt + answer），从配置中读取
MAX_LEN = ModelConfig.MAX_TEXT_LEN


# --------------------------------
# 1. 从原始数据中抽取关键信息
# --------------------------------
def extract_fields(example, is_eval: bool = False):
    """
    从原始 ChemCot 或 ChemLLMBench 数据中提取：
    - query: 作为 prompt
    - input_smiles: 分子 SMILES（用于多模态分子编码器）
    - label: 作为 LLM 的监督答案

    当 is_eval=True 时，label / cot / cot_steps 将被置为 None（例如用于推理/评估）。
    """

    # meta 字段 might be a JSON string or already a dictionary
    if isinstance(example.get("meta", ""), str):
        meta_dict = json.loads(example["meta"]) if example.get("meta") else {}
    else:
        meta_dict = example.get("meta", {})

    task = example.get('subtask', example.get('task', 'unknown'))

    # Check if this is ChemLLMBench format (no struct_cot) or ChemCot format (has struct_cot)
    has_struct_cot = "struct_cot" in example and example["struct_cot"] is not None

    if is_eval:
        # For eval mode, we don't process struct_cot regardless of format
        cot_dict = {}
        cot_steps = []
        cot_value = ""

        # Extract label for both formats in eval mode (will be None anyway)
        label_value = ""
    elif has_struct_cot:
        # This is ChemCot format
        try:
            cot_content = json.loads(example["struct_cot"], object_pairs_hook=OrderedDict)
        except json.JSONDecodeError as e:
            print(f"\n[CRITICAL DATA ERROR] JSON is malformed in example ID: {example.get('id')}")
            print(f"[ERROR DETAILS]: {e}")
            print(f"[RAW CONTENT]: {repr(example.get('struct_cot'))}")
            raise

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
                raise
        else:
            cot_dict = cot_content

        # 3. 构造 CoT 步骤列表 (Coconut 专用)
        # 每个步骤是一个字符串，例如 "Step 1:\nSMILES: CCC"
        cot_steps = []
        for i, (k, v) in enumerate(cot_dict.items()):
            if k == "output":
                continue
            cot_steps.append(f"Step {i+1}:\n{k}: {v}")

        cot_value = "\n\n".join(cot_steps)

        # 4. 提取 label 优先级 for ChemCot
        if meta_dict.get("gt"):
            label_value = str(meta_dict["gt"])
        elif meta_dict.get("reference"):
            label_value = str(meta_dict["reference"])
        else:
            label_value = str(cot_dict.get("output", ""))
    else:
        # This is ChemLLMBench format
        cot_dict = {}
        cot_steps = []
        cot_value = ""

        # Extract label from gt field or reference in meta
        if example.get("gt") and example.get("gt").strip() != "":
            label_value = str(example["gt"])
        elif meta_dict.get("reference"):
            label_value = str(meta_dict["reference"])
        elif meta_dict.get("molecule"):
            # For molecule captioning tasks
            label_value = str(meta_dict["molecule"])
        else:
            label_value = ""

    # 提取 SMILES - handle both ChemCot and ChemLLMBench formats
    raw_val = meta_dict.get("molecule")
    # Check if raw_val is None or empty string
    if not raw_val:  # This checks for None, empty string, empty list, etc.
        # Try both singular and plural forms to handle different data formats
        reactant = meta_dict.get("reactant") or meta_dict.get("reactants") or []
        reagent = meta_dict.get("reagent") or meta_dict.get("reagents") or []
        product = meta_dict.get("product") or meta_dict.get("products") or []

        # Handle both string and list formats
        if isinstance(reactant, str):
            reactant = [reactant] if reactant else []
        if isinstance(reagent, str):
            reagent = [reagent] if reagent else []
        if isinstance(product, str):
            product = [product] if product else []

        raw_val = reactant + reagent + product

    if isinstance(raw_val, str):
        val_list = [raw_val]
    elif isinstance(raw_val, list):
        val_list = raw_val
    else:
        val_list = []
        if example.get('id'):
            print(example['id'])

    # 按 '.' 切分并处理末尾点的情况，同时保持顺序
    input_smiles = []
    for s in val_list:
        if isinstance(s, str):
            # split('.') 会把 "C.C." 变成 ["C", "C", ""]
            # 通过 if part 过滤掉空字符串，正好相当于去掉了末尾的点或连续的点
            for part in s.split('.'):
                if part and part != '[]':  # Also filter out '[]' strings that may come from empty lists
                    input_smiles.append(part)

    # Additional filtering to remove any empty strings or '[]' that might have slipped through
    input_smiles = [smile for smile in input_smiles if smile and smile != '[]' and smile.strip()]

    # Process candidate_rank if it exists in meta
    candidate_rank_str = meta_dict.get("candidate_rank")
    print(candidate_rank_str)
    if candidate_rank_str:
        try:
            # candidate_rank_list = ast.literal_eval(candidate_rank_str)
            input_smiles.append(candidate_rank_str)
            # if isinstance(candidate_rank_list, list):
            #     # Shuffle the candidate rank list
            #     shuffled_candidate_rank = candidate_rank_list.copy()
            #     random.shuffle(shuffled_candidate_rank)

            #     # Add shuffled candidates to input_smiles if not already present
            #     temp=[]
            #     for candidate in shuffled_candidate_rank:
            #         if candidate not in input_smiles and isinstance(candidate, str) and candidate and candidate != '[]':
            #             # temp.append(candidate)
            #             pass
                
        except (ValueError, SyntaxError):
            # If literal_eval fails, skip processing this field
            pass

    # 处理 query 中的 SMILES 标记
    query = example.get("query", "")

    # --------------------------------
    # 替换特定的 JSON 格式要求为 <answer> 格式 (only for non-eval mode)
    # --------------------------------
    if query and not is_eval:
        # Apply transformations only for ChemCot format (which has specific formatting requirements)
        if has_struct_cot:
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
        else:
            # For ChemLLMBench format, check if the query already contains <answer> format
            # If not, we might need to add it depending on the task
            if "<answer>" not in query:
                # For most ChemLLMBench tasks, we want to ensure the answer format is specified
                if "final answer" not in query.lower():
                    query = query.rstrip() + "\nYour final answer must be formatted as <answer> Your Answer </answer>"

        query = query.strip()

    if query and input_smiles:
        # Split query by answer markers to avoid extracting SMILES from answer sections
        # Find the position of <answer> tag to only process the query part before it
        answer_pos = query.find('<answer>')
        if answer_pos != -1:
            # Only process the part of query before the answer
            query_before_answer = query[:answer_pos]
            query_after_answer = query[answer_pos:]
        else:
            # No answer tag found, process the whole query
            query_before_answer = query
            query_after_answer = ""

        # 1. 按长度从长到短排序，防止短 SMILES (如 C) 误匹配长 SMILES (如 CC) 的一部分
        indexed_smiles = sorted(enumerate(input_smiles), key=lambda x: len(x[1]), reverse=True)

        # 边界检查字符集：防止误伤单词（Cat）或长链内部（C1...）
        smiles_chars = r'a-zA-Z0-9\[\]\(\)\=#@+\-\/\\%'

        for i, s in indexed_smiles:
            # 使用正则进行边界检查，确保匹配的是独立的 SMILES 实体
            pattern = rf'(?<![{smiles_chars}]){re.escape(s)}(?![{smiles_chars}])'
            if re.search(pattern, query_before_answer):
                # 使用 lambda 替换，避免 re.sub 对 SMILES 中反斜杠 (\) 的错误转义
                replacement = f"{s} (the {i+1}-th SMILES)"
                query_before_answer = re.sub(pattern, lambda m: replacement, query_before_answer)

        # Combine the processed query part with the answer part
        query = query_before_answer + query_after_answer

    # 如果是 eval 模式，将这些监督/中间字段设为 None
    if is_eval:
        return {
            "query": query,
            "input_smiles": input_smiles or [""],
            "label": None,
            "cot": None,
            "cot_steps": None,
            "task": task
        }

    return {
        # LLM 输入的文本 prompt
        "query": query,
        # 分子 SMILES 列表
        "input_smiles": input_smiles or [""],
        # LLM 的监督答案
        "label": f"<answer> {label_value} </answer>" if label_value else None,
        # 结构化思维链 (CoT)
        "cot": cot_value,
        # 分步思维链 (Coconut 专用)
        "cot_steps": cot_steps,
        "task": task
    }


# --------------------------------
# 2.5 构造 Coconut 训练样本
# --------------------------------
def coconut_tokenize(
    example,
    scheduled_stage=0,
    c_thought=2,
    max_len=ModelConfig.MAX_TEXT_LEN,
    is_eval: bool = False,
):
    """
    Coconut 训练的核心数据处理。

    当 is_eval=True 时，只返回 prompt 的 tokenized 结果，并把 labels 设为 None（用于评估/推理）。
    """
    # Get tokenizer and constants
    tokenizer, COCONUT_TOKENS, LATENT_ID, START_LATENT_ID, END_LATENT_ID = get_tokenizer_constants()

    prompt = example.get("query", "")
    steps = example.get("cot_steps") or []
    label = example.get("label")

    # Eval 模式：只 token 化 prompt
    if is_eval or (label is None and not steps):
        prompt_ids = tokenizer.encode(f"{prompt}\n\n", add_special_tokens=False)
        input_ids = prompt_ids[:max_len]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": None,
            "smiles": example.get("input_smiles"),
        }

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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "smiles": example.get("input_smiles"),
    }


# --------------------------------
# 2. 构造 Causal LM 的训练样本 (旧版 SFT 兼容)
# --------------------------------
def llm_tokenize(example, include_cot=True, max_len=ModelConfig.MAX_TEXT_LEN, is_eval: bool = False):
    """
    构造 Causal Language Model 的训练格式。

    当 is_eval=True 或 label 为 None 时，仅 token 化 prompt，并将 labels 设为 None。
    """
    # Get tokenizer
    tokenizer, _, _, _, _ = get_tokenizer_constants()

    prompt = example.get("query", "")
    cot = example.get("cot") or ""
    label = example.get("label")

    if is_eval or (label is None):
        # 仅 prompt
        prompt_enc = tokenizer(f"{prompt}\n\n", truncation=True, padding=False, max_length=max_len, add_special_tokens=False)
        input_ids = prompt_enc["input_ids"][:max_len]
        attention_mask = prompt_enc["attention_mask"][:max_len]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": None,
            "smiles": example.get("input_smiles") or "",
        }

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

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "smiles": example.get("input_smiles") or "",
    }


# --------------------------------
# 3. 数据集加载与整体处理流程（支持 eval_mode）
# --------------------------------
def load_data(
    path,
    include_cot=True,
    max_len=ModelConfig.MAX_TEXT_LEN,
    is_coconut=False,
    scheduled_stage=0,
    c_thought=2,
    exclude_tasks=['rcr'],
    eval_mode: bool = False,
):
    """
    完整的数据加载流程，新增参数 eval_mode：
      - eval_mode=False（默认）：训练/微调用，有 label 和 cot 信息
      - eval_mode=True：评估/推理用，label/cot/cot_steps 被置为 None，tokenize 时不会产生 labels
    """
    all_json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)

    def filter_data(f):
        return all([not f.endswith(f"{task}.json") for task in exclude_tasks])

    data_files = [f for f in all_json_files if filter_data(f)]

    # Load each JSON file manually to avoid caching issues and handle schema differences
    import json
    from datasets import Dataset
    datasets_list = []

    for file_path in data_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Normalize the data to ensure consistent schema across all files (both ChemCot and ChemLLMBench)
        normalized_data = []
        for item in data:
            # Ensure all expected fields exist to maintain consistent schema
            # Handle both ChemCot and ChemLLMBench formats

            # Normalize the 'meta' field - preserve all original keys, not just predefined ones
            meta = item.get('meta', {})
            if not isinstance(meta, dict):
                meta = {'molecule': meta} if meta is not None else {}

            # Create a new meta dict that includes all original keys plus defaults for missing expected keys
            normalized_meta = {}

            # First, copy all original keys
            for key, value in meta.items():
                if isinstance(value, list):
                    # Ensure list elements are strings
                    if isinstance(value, list):
                        normalized_meta[key] = [str(v) for v in value if v is not None]
                    elif value is None:
                        normalized_meta[key] = []
                    else:
                        normalized_meta[key] = [str(value)] if str(value) != '' else []
                else:  # string or other type
                    normalized_meta[key] = str(value) if value is not None else ''

            # Then ensure expected keys exist with default values if not already present
            expected_meta_keys = {
                'molecule': '',
                'reactants': [],
                'reagents': [],
                'products': [],
                'candidate_rank': '',
                'reference': ''
            }

            for key, default_value in expected_meta_keys.items():
                if key not in normalized_meta:
                    normalized_meta[key] = default_value

            # Create normalized item with all required fields
            normalized_item = {
                'id': item.get('id', ''),
                'query': item.get('query', ''),
                'struct_cot': item.get('struct_cot', None),  # May be None for ChemLLMBench
                'subtask': item.get('subtask', item.get('task', 'unknown')),
                'task': item.get('task', item.get('subtask', 'unknown')),
                'gt': item.get('gt', ''),
                'raw_cot': item.get('raw_cot', ''),
                'cot_result': item.get('cot_result', ''),
                'meta': normalized_meta
            }

            normalized_data.append(normalized_item)

        # Convert to HuggingFace Dataset
        if normalized_data:  # Only create dataset if there's data
            single_ds = Dataset.from_list(normalized_data)
            datasets_list.append(single_ds)

    # Check if datasets_list is empty before concatenating
    if not datasets_list:
        print(f"Warning: No datasets found in {path} with the specified exclude_tasks {exclude_tasks}. Returning empty dataset.")
        # Return an empty dataset with the expected schema
        from datasets import Dataset
        empty_dataset = Dataset.from_dict({
            "id": [],
            "query": [],
            "struct_cot": [],
            "meta": [],
            "subtask": [],
            "task": [],
            "gt": [],
            "raw_cot": [],
            "cot_result": []
        })
        ds = empty_dataset
    else:
        # Ensure all datasets have the same schema before concatenating
        # Cast all datasets to have consistent schema
        from datasets import Dataset, Features, Value, Sequence
        # Define a consistent schema for all datasets (supporting both ChemCot and ChemLLMBench)
        # Include all possible meta fields that might appear in either format
        consistent_schema = Features({
            'id': Value('string'),
            'query': Value('string'),
            'struct_cot': Value('string'),  # Can be null for ChemLLMBench
            'raw_cot': Value('string'),
            'subtask': Value('string'),
            'task': Value('string'),
            'gt': Value('string'),
            'cot_result': Value('string'),
            'meta': {
                'candidate_rank': Value('string'),
                'molecule': Value('string'),
                'reactant': Value('string'),  # Added for ChemLLMBench
                'reactants': Sequence(Value('string')),
                'reagent': Value('string'),   # Added for ChemLLMBench
                'reagents': Sequence(Value('string')),
                'product': Value('string'),   # Added for ChemLLMBench
                'products': Sequence(Value('string')),
                'reference': Value('string')
            }
        })

        # Reformat each dataset to have the consistent schema
        reformatted_datasets = []
        for ds in datasets_list:
            # Convert dataset to pandas, then reconstruct with consistent schema
            try:
                df = ds.to_pandas()

                # Process the meta column to ensure consistent types for all rows
                def process_meta_row(meta_dict):
                    # Make a copy to avoid modifying the original
                    processed_meta = meta_dict.copy() if isinstance(meta_dict, dict) else {}

                    # Process list fields to ensure they are proper lists of strings
                    list_fields = ['reactants', 'reagents', 'products']
                    for field in list_fields:
                        if field in processed_meta:
                            value = processed_meta[field]
                            if value is None:
                                processed_meta[field] = []
                            elif not isinstance(value, list):
                                # If it's not a list, try to convert it
                                if isinstance(value, str):
                                    processed_meta[field] = [value] if value else []
                                else:
                                    processed_meta[field] = [str(value)] if str(value) else []
                            else:
                                # It's already a list, ensure all elements are strings and not None
                                processed_meta[field] = [str(item) for item in value if item is not None]
                        else:
                            # Field doesn't exist, add it with default empty list
                            processed_meta[field] = []

                    # Process string fields to ensure they are strings
                    string_fields = ['molecule', 'candidate_rank', 'reference', 'reactant', 'reagent', 'product']
                    for field in string_fields:
                        if field in processed_meta:
                            value = processed_meta[field]
                            processed_meta[field] = str(value) if value is not None else ''
                        else:
                            # Field doesn't exist, add it with default empty string
                            processed_meta[field] = ''

                    return processed_meta

                # Apply the processing function to all meta entries
                df['meta'] = df['meta'].apply(process_meta_row)

                # Create new dataset with consistent schema
                reformatted_ds = Dataset.from_pandas(df, features=consistent_schema)
                reformatted_datasets.append(reformatted_ds)
            except Exception as e:
                print(f"Error processing dataset: {e}")
                raise

        # Now concatenate the reformatted datasets
        from datasets import concatenate_datasets
        ds = concatenate_datasets(reformatted_datasets)

    bad_ids = [
        "f7e567a6-47de-4c77-8c1f-9049689322e8",
        "bedfe3e8-ab07-4b8e-b872-ae281e5f55af",
        "9cb0a77d-6203-4686-9c8b-45fd3fc770f2"
    ]
    ds = ds.filter(lambda x: "id" not in x or x["id"] not in bad_ids)

    # Step 1: 提取结构化字段（支持 eval 模式）
    dataset = ds.map(
        extract_fields,
        batched=False,
        fn_kwargs={"is_eval": eval_mode},
        remove_columns=ds.column_names
    )

    # Step 2: 构造训练/评估样本
    if is_coconut:
        dataset = dataset.map(
            coconut_tokenize,
            batched=False,
            fn_kwargs={
                "scheduled_stage": scheduled_stage,
                "c_thought": c_thought,
                "max_len": max_len,
                "is_eval": eval_mode,
            },
            remove_columns=["query", "input_smiles", "label", "cot", "cot_steps"]
        )
    else:
        dataset = dataset.map(
            llm_tokenize,
            batched=False,
            fn_kwargs={"include_cot": include_cot, "max_len": max_len, "is_eval": eval_mode},
            remove_columns=["query", "input_smiles", "label", "cot", "cot_steps"]
        )

    return dataset


# --------------------------------
# 4. 运行示例与测试
# --------------------------------
if __name__ == "__main__":
    # DATA_ROOT = "/mnt/afs/L202500070/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot"

    # # 示例：训练集加载（含 labels）
    # ds_train = load_data(DATA_ROOT, include_cot=True, is_coconut=False, eval_mode=False)
    # print(f"Train samples example: {ds_train[0]}")

    EVAL_DATA = '/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/data/ChemLLMBench/chemllmbench'
    # 示例：eval 集合加载（label/cot/cot_steps 为 None，tokenize 时不会产生 labels）,exclude_tasks=['molecule_captioning','molecule_design','name_prediction','reaction_prediction','retro']
    ds_eval = load_data(EVAL_DATA, include_cot=False, is_coconut=False, exclude_tasks=['molecule_captioning','molecule_design','name_prediction','reaction_prediction','retro'],eval_mode=True)#,exclude_tasks=['molecule_captioning','molecule_design','name_prediction','reaction_prediction','retro'],
    print(f"Eval samples example: {ds_eval[0]}")