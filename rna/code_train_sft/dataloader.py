# ============================
# Part 1. Dataset loading & preprocessing
# ============================

import json
import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

# --------------------------------
# Load tokenizer (Qwen decoder-only LM)
# --------------------------------
# 使用 Qwen4B 的 tokenizer
# 对于 decoder-only 模型，通常将 pad_token 设置为 eos_token
tokenizer = AutoTokenizer.from_pretrained(
    "/zengdaojian/zhangjia/BioLatent/Qwen4B"
)
tokenizer.pad_token = tokenizer.eos_token

# 最大文本长度（prompt + answer）
MAX_LEN = 128


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
    meta_dict = json.loads(example["meta"])

    # label 优先级选择
    if meta_dict.get("gt"):
        label_value = str(meta_dict["gt"])
    elif meta_dict.get("reference"):
        label_value = str(meta_dict["reference"])
    else:
        # 从 struct_cot 中用正则提取 "output": "xxx"
        struct_cot = example.get("struct_cot", "")
        match = re.search(r'"output"\s*:\s*"(\w+)"', struct_cot)
        label_value = match.group(1) if match else ""

    return {
        # LLM 输入的文本 prompt
        "query": example.get("query", ""),
        # 分子 SMILES，去掉可能存在的 '.'（多片段）
        "input_smiles": meta_dict.get("molecule", "C").replace(".", ""),
        # LLM 的监督答案
        "label": label_value,
    }


# --------------------------------
# 2. 构造 Causal LM 的训练样本
# --------------------------------
def llm_tokenize(example):
    """
    构造 Causal Language Model 的训练格式：

        [PROMPT] <eos> [ANSWER]

    训练目标：
        - 只在 ANSWER 部分计算 loss
        - PROMPT 部分的 label 设为 -100
    """

    prompt = example["query"]
    answer = example["label"]

    # prompt 与 answer 用 eos_token 分隔
    full_text = prompt + tokenizer.eos_token + answer

    # 对完整文本进行 tokenization
    enc = tokenizer(
        full_text,
        truncation=True,              # 超长截断
        padding="max_length",         # padding 到 MAX_LEN
        max_length=MAX_LEN,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # -------- 构造 labels --------
    # 初始 labels 与 input_ids 相同
    labels = input_ids.copy()

    # 单独对 prompt + eos 进行 tokenize，用来确定 prompt 的 token 长度
    prompt_ids = tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=MAX_LEN,
    )["input_ids"]

    prompt_len = len(prompt_ids)

    # 将 prompt 部分的 label mask 掉（不计算 loss）
    labels[:prompt_len] = [-100] * prompt_len

    # debug：打印当前样本的 SMILES（注意：数据量大时应删除）
    print(example["input_smiles"])

    return {
        # LLM 的输入 token
        "input_ids": input_ids,
        # attention mask
        "attention_mask": attention_mask,
        # Causal LM 的监督信号（prompt 部分为 -100）
        "labels": labels,
        # 分子 SMILES（供 Qwen3MoleculeLLM 的 forward 使用）
        "smiles": example["input_smiles"],
    }


# --------------------------------
# 3. 数据集加载与整体处理流程
# --------------------------------
def load_data(path):
    """
    完整的数据加载流程：
    1. 加载原始 ChemCot 数据集
    2. 提取 query / smiles / label
    3. 将文本转为 LLM 可训练的 token 格式
    """

    # 加载 HuggingFace datasets 格式的数据
    ds = load_dataset("/zengdaojian/zhangjia/BioLatent/ChemCotDataset")["train"]

    print("Raw dataset example:")
    # print(ds[0])  # 可用于调试原始格式

    # --------------------------------
    # Step 1: 提取结构化字段
    # --------------------------------
    dataset = ds.map(
        extract_fields,
        batched=False,
        remove_columns=ds.column_names  # 移除原始无关字段
    )

    # print("After extract_fields:")
    # print(dataset[0])

    # --------------------------------
    # Step 2: 构造 LLM 训练样本
    # --------------------------------
    dataset = dataset.map(
        llm_tokenize,
        batched=False,
        remove_columns=["query", "label", "input_smiles"]
    )

    return dataset


# --------------------------------
# 4. 运行示例
# --------------------------------
dataset = load_data("/zengdaojian/zhangjia/BioLatent/ChemCotDataset/chemcotbench-cot")

print("Final tokenized dataset example:")
print(dataset[0])
