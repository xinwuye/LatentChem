import os

class ModelConfig:
    # --- 模型架构参数 (Architecture) ---
    INPUT_DIM = 768      # 分子编码器输出维度 (SMI-TED Light)
    NUM_QUERIES = 128    # 投影器的查询数量
    # OUTPUT_DIM 会在 model_new.py 中自动根据 LLM 维度适配
    NUM_HEADS = 8
    
    # --- 数据对齐参数 (Alignment) ---
    # 分子前缀总长度 = num_queries + <mol_start> + <mol_end>
    # 默认 128 + 2 = 130
    SMILES_LEN = NUM_QUERIES + 2 
    
    # --- 数据加载参数 (Data Loading) ---
    MAX_TEXT_LEN = 8192   # 文本部分的最大长度 (dataloader.py 中使用)
    
    # --- 默认路径 (Default Paths) ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Qwen 模型路径
    DEFAULT_QWEN_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../models/Qwen3-8B-Base"))
    
    # SMI-TED 模型文件夹和权重文件名
    DEFAULT_SMI_TED_FOLDER = os.path.abspath(os.path.join(CURRENT_DIR, "../models/smi-ted"))
    DEFAULT_SMI_TED_CKPT = "smi-ted-Light_40.pt"
    
    # 数据集路径
    DEFAULT_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../ChemCotDataset/chemcotbench-cot"))
    TEST_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../ChemCotDataset/chemcotbench"))
    
    # 训练输出目录
    DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "qwen3_mol_sft_lora_results")
    TEST_DATA_PATH = os.path.join(CURRENT_DIR, "qwen3_mol_sft_lora_test_results")

