import os

class ModelConfig:
    # --- 模型架构参数 (Architecture) ---
    INPUT_DIM = 640      # RNA编码器输出维度 (RNA-FM)
    NUM_QUERIES = 128    # 投影器的查询数量
    # OUTPUT_DIM 会在 model_new.py 中自动根据 LLM 维度适配
    NUM_HEADS = 8
    
    # --- 数据对齐参数 (Alignment) ---
    # 分子前缀总长度 = num_queries + <mol_start> + <mol_end>
    # 默认 128 + 2 = 130
    RNA_LEN = NUM_QUERIES + 2 
    
    # --- 数据加载参数 (Data Loading) ---
    MAX_TEXT_LEN = 512   # 文本部分的最大长度 (dataloader.py 中使用)
    
    # --- 默认路径 (Default Paths) ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Qwen 模型路径
    DEFAULT_QWEN_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "models", "Qwen3-8B-Base"))
    
    # 数据集路径
    DEFAULT_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../extracted/stage_3/rna_stage3_with_gpt_responses.csv"))
    
    # RNA Embedding Path
    RNA_EMBEDDING_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../extracted/stage_3/biolatentcot_stage3"))

    # 训练输出目录
    DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "qwen3_mol_sft_lora_results")

