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

    DEFAULT_QWEN_SIZE = "8b"

    QWEN_SPECS = {
        "0.6b": {
            "hf_repo": "Qwen/Qwen3-0.6B-Base",
            "local_dir": os.path.abspath(os.path.join(CURRENT_DIR, "../models/Qwen3-0.6B-Base")),
            "display_name": "Qwen3-0.6B-Base",
        },
        "1.7b": {
            "hf_repo": "Qwen/Qwen3-1.7B-Base",
            "local_dir": os.path.abspath(os.path.join(CURRENT_DIR, "../models/Qwen3-1.7B-Base")),
            "display_name": "Qwen3-1.7B-Base",
        },
        "4b": {
            "hf_repo": "Qwen/Qwen3-4B-Base",
            "local_dir": os.path.abspath(os.path.join(CURRENT_DIR, "../models/Qwen3-4B-Base")),
            "display_name": "Qwen3-4B-Base",
        },
        "8b": {
            "hf_repo": "Qwen/Qwen3-8B-Base",
            "local_dir": os.path.abspath(os.path.join(CURRENT_DIR, "../models/Qwen3-8B-Base")),
            "display_name": "Qwen3-8B-Base",
        },
        "14b": {
            "hf_repo": "Qwen/Qwen3-14B-Base",
            "local_dir": os.path.abspath(os.path.join(CURRENT_DIR, "../models/Qwen3-14B-Base")),
            "display_name": "Qwen3-14B-Base",
        },
    }
    
    # Qwen 模型路径
    DEFAULT_QWEN_PATH = QWEN_SPECS[DEFAULT_QWEN_SIZE]["local_dir"]
    
    # SMI-TED 模型文件夹和权重文件名
    DEFAULT_SMI_TED_FOLDER = os.path.abspath(os.path.join(CURRENT_DIR, "../models/smi-ted"))
    DEFAULT_SMI_TED_CKPT = "smi-ted-Light_40.pt"
    
    # 数据集路径
    DEFAULT_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../ChemCotDataset/chemcotbench-cot"))
    TEST_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../ChemCotDataset/chemcotbench"))
    
    # 训练输出目录
    DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "qwen3_mol_sft_lora_results")
    TEST_DATA_PATH = os.path.join(CURRENT_DIR, "qwen3_mol_sft_lora_test_results")

    @classmethod
    def normalize_qwen_size(cls, size):
        if size is None:
            return cls.DEFAULT_QWEN_SIZE
        normalized = str(size).strip().lower()
        if normalized not in cls.QWEN_SPECS:
            allowed = ", ".join(sorted(cls.QWEN_SPECS.keys()))
            raise ValueError(f"Unsupported qwen size: {size!r}. Allowed values: {allowed}")
        return normalized

    @classmethod
    def get_qwen_spec(cls, size):
        normalized = cls.normalize_qwen_size(size)
        return cls.QWEN_SPECS[normalized]

    @classmethod
    def get_qwen_path(cls, size):
        return cls.get_qwen_spec(size)["local_dir"]

    @classmethod
    def get_qwen_hf_repo(cls, size):
        return cls.get_qwen_spec(size)["hf_repo"]

    @classmethod
    def get_qwen_display_name(cls, size):
        return cls.get_qwen_spec(size)["display_name"]

    @classmethod
    def require_qwen_path(cls, size):
        normalized = cls.normalize_qwen_size(size)
        path = cls.get_qwen_path(normalized)
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Qwen model directory for --qwen_size {normalized!r} does not exist: {path}"
            )
        return path
