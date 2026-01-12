# ============================
# Part 1. Dataset loading & preprocessing (with eval mode)
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

# --------------------------------
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


# --------------------------------
# 1. 从原始数据中抽取关键信息
# --------------------------------
def extract_fields(example, is_eval: bool = False):
    """
    从原始 ChemCot 数据中提取：
    - query: 作为 prompt
    - input_smiles: 分子 SMILES（用于多模态分子编码器）
    - label: 作为 LLM 的监督答案

    当 is_eval=True 时，label / cot / cot_steps 将被置为 None（例如用于推理/评估）。
    """
    
    # meta 字段是一个 JSON 字符串，需要先解析
    meta_dict = json.loads(example["meta"])
    task = example['subtask']

    # 2. 解析 struct_cot
    # 如果 struct_cot 是不可解析的 JSON 会抛出错误并让上层决定
    if is_eval:
        cot_dict = {}
    else:
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
        reactants = meta_dict.get("reactants", [])
        reagents = meta_dict.get("reagents", [])
        products = meta_dict.get("products", [])
        raw_val = reactants + reagents + products

    if isinstance(raw_val, str):
        val_list = [raw_val]
    elif isinstance(raw_val, list):
        val_list = raw_val
    else:
        val_list = []
        print(example['id'])
        
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
    if query and not is_eval:
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

    # 如果是 eval 模式，将这些监督/中间字段设为 None
    if is_eval:
        return {
            "query": query,
            "input_smiles": input_smiles,
            "label": None,
            "cot": None,
            "cot_steps": None,
            "task": task
        }

    return {
        # LLM 输入的文本 prompt
        "query": query,
        # 分子 SMILES 列表
        "input_smiles": input_smiles,
        # LLM 的监督答案
        "label": f"<answer> {label_value} </answer>",
        # 结构化思维链 (CoT)
        "cot": cot_value,
        # CoT 字符长度（用于动态分配 latent 数量）
        "cot_len": len(cot_value) if cot_value is not None else 0,
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
            "smiles": example.get("input_smiles"),
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
        "smiles": example.get("input_smiles"),
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

    ds = load_dataset("json", data_files=data_files)["train"]

    bad_ids = [
        "f7e567a6-47de-4c77-8c1f-9049689322e8",
        "bedfe3e8-ab07-4b8e-b872-ae281e5f55af",
        "9cb0a77d-6203-4686-9c8b-45fd3fc770f2"
    ]
    ds = ds.filter(lambda x: x["id"] not in bad_ids)

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
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in ("prompt", "input_smiles", "label")])
    return dataset


# ============================
# Part 2. InstructMol Dataset Loading (Molecule-oriented Instructions)
# ============================

# --------------------------------
# 1. Extract fields from InstructMol format
# --------------------------------
def extract_instructmol_fields(example, is_eval: bool = False):
    """
    Extract fields from InstructMol Molecule-oriented_Instructions dataset.

    Format:
    - instruction: The task instruction
    - input: SMILES string(s) (may contain multiple molecules separated by '.')
    - output: The expected answer
    - metadata: Dict containing task and split info

    Returns:
    - query: instruction as prompt
    - input_smiles: list of SMILES strings
    - label: formatted answer (None if is_eval=True)
    - task: task name from metadata
    """
    instruction = example.get("instruction", "")
    input_smiles_str = example.get("input", "")
    output = example.get("output", "")
    metadata = example.get("metadata", {})
    task = metadata.get("task", "unknown")

    # Parse SMILES input - split by '.' and remove empty strings
    input_smiles = []
    if input_smiles_str:
        # Split by '.' and filter out empty strings
        for part in input_smiles_str.split('.'):
            if part.strip():
                input_smiles.append(part.strip())

    # Format the query (instruction as prompt)
    query = instruction.strip()

    # Add answer formatting instruction
    if not is_eval:
        query = query.rstrip() + "\nYour final answer must be formatted as <answer> Your Answer </answer>"

    # Label formatting
    if is_eval:
        label_value = None
    else:
        # Convert output to string and wrap in answer tags
        label_value = f"<answer> {str(output)} </answer>"

    return {
        "query": query,
        "input_smiles": input_smiles,
        "label": label_value,
        "cot": None,  # InstructMol doesn't have CoT
        "cot_steps": None,  # InstructMol doesn't have CoT steps
        "task": task
    }


# --------------------------------
# 2. Load InstructMol dataset
# --------------------------------
def load_instructmol_data(
    path,
    max_len=ModelConfig.MAX_TEXT_LEN,
    is_coconut=False,
    scheduled_stage=0,
    c_thought=2,
    eval_mode: bool = False,
):
    """
    Load InstructMol Molecule-oriented_Instructions dataset.

    Args:
        path: Path to the directory containing InstructMol JSON files
        max_len: Maximum sequence length
        is_coconut: Whether to use Coconut tokenization (not applicable for InstructMol)
        scheduled_stage: Coconut stage (not applicable for InstructMol)
        c_thought: Number of latent tokens per CoT step (not applicable for InstructMol)
        eval_mode: If True, don't include labels (for inference)

    Returns:
        HuggingFace Dataset with tokenized inputs
    """
    all_json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)

    if not all_json_files:
        raise ValueError(f"No JSON files found in {path}")

    # Load the dataset
    ds = load_dataset("json", data_files=all_json_files)["train"]

    # Step 1: Extract structured fields
    dataset = ds.map(
        extract_instructmol_fields,
        batched=False,
        fn_kwargs={"is_eval": eval_mode},
        remove_columns=ds.column_names
    )

    # Step 2: Tokenize (use llm_tokenize since InstructMol has no CoT)
    # Note: InstructMol doesn't have CoT, so we always use include_cot=False
    dataset = dataset.map(
        llm_tokenize,
        batched=False,
        fn_kwargs={"include_cot": False, "max_len": max_len, "is_eval": eval_mode},
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

    EVAL_DATA = '../data/ChemCoTBench'
    # 示例：eval 集合加载（label/cot/cot_steps 为 None，tokenize 时不会产生 labels）
    ds_eval = load_data(EVAL_DATA, include_cot=False, is_coconut=False, eval_mode=True, exclude_tasks=['rcr', 'mechsel'])
    print(f"Eval samples example: {ds_eval[0]}")
