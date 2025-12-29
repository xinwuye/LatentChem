import json
import re
import torch
import pandas as pd
import glob
import numpy as np
import os
from datasets import Dataset
from transformers import AutoTokenizer

# --------------------------------
# Load tokenizer
# --------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "/home/bingxing2/ailab/fanyutao/biolatentcot/yoyo/BioLatentCOT/rna/code_train_sft/models/Qwen3-8B-Base"
)
tokenizer.pad_token = tokenizer.eos_token
MAX_LEN = 128

# --------------------------------
# 1. Load the RNA Representations
# --------------------------------
# We load this globally so the map function can access it

# 1. Initialize an empty master dictionary
all_rna_representations = {}

# 2. Find all .npy files in all subdirectories (recursive search)
# The "**" pattern looks through every folder level
npy_files = glob.glob('/home/bingxing2/ailab/fanyutao/biolatentcot/yoyo/RNA-FM/redevelop/results/biolatentcot_stage3/**/*.npy', recursive=True)

print(f"Found {len(npy_files)} .npy files. Loading...")

for file_path in npy_files:
    # Skip log files or files that aren't the collections
    if "representations-collection" in file_path:
        data = np.load(file_path, allow_pickle=True).item()
        # Merge this file's dictionary into our master dictionary
        all_rna_representations.update(data)

print(f"Total unique RNA representations loaded: {len(all_rna_representations)}")

# --------------------------------
# 2. Extract fields from CSV row
# --------------------------------
def extract_csv_fields(example, idx):
    """
    example: a row from the CSV
    idx: the index of the row (to match npy keys)
    """
    # 1. The prompt is in the 'input' column
    query = example.get("input", "")
    
    # 2. Get the label (answer) -> this is not super 
    label_value = str(example.get("label", ""))

    # 3. Get RNA representation from the npy dictionary using the index
    rna_latent = all_rna_representations.get(idx, None)
    
    # If the key is a string in the npy, use rna_repr_dict.get(str(idx))
    
    return {
        "query": query,
        "label": label_value,
        "rna_latent": rna_latent, # This is the (96, 640) array
        "rna_seq": example.get("extracted_sequences", "")
    }

# --------------------------------
# 3. Tokenize and Format for LLM
# --------------------------------
def llm_tokenize(example):
    prompt = example["query"]
    answer = example["label"]

    full_text = prompt + tokenizer.eos_token + answer

    enc = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Labels for Causal LM (-100 for prompt)
    labels = input_ids.copy()
    prompt_ids = tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=MAX_LEN,
    )["input_ids"]
    prompt_len = len(prompt_ids)
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "rna_latent": example["rna_latent"], # Passed through to trainer
    }

# --------------------------------
# 4. Modified Data Loading Flow
# --------------------------------
def load_data(csv_path):
    # 1. Load CSV using pandas
    df = pd.read_csv(csv_path)
    
    # Optional: Fix the "Unnamed" column if it represents the true index
    # If the npy keys match the 'Unnamed: 0' column specifically:
    # df = df.set_index('Unnamed: 0') 

    # 2. Convert to HuggingFace Dataset
    raw_ds = Dataset.from_pandas(df)

    # 3. Map the extraction (includes npy lookup)
    # with_indices=True allows us to access the row index to match the npy key
    dataset = raw_ds.map(
        extract_csv_fields,
        with_indices=True,
        remove_columns=raw_ds.column_names
    )

    # 4. Tokenize
    dataset = dataset.map(
        llm_tokenize,
        remove_columns=["query", "label", "rna_seq"]
    )

    return dataset

# --------------------------------
# 5. Execution
# --------------------------------
dataset = load_data("/home/bingxing2/ailab/fanyutao/biolatentcot/yoyo/BioLatentCOT/rna/extracted/stage3/rna_stage3.csv")

print("\nFinal processed sample:")
print(f"Input IDs Length: {len(dataset[0]['input_ids'])}")
print(f"RNA Latent Shape: {np.array(dataset[0]['rna_latent']).shape}")