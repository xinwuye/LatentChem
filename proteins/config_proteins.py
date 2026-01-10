import os

class ModelConfig:
    # ==================================================
    # Model architecture parameters
    # ==================================================

    # Dimension of the protein encoder output
    # ESM-2 650M produces 1280-dimensional embeddings
    INPUT_DIM = 1280

    # Number of learnable query vectors used by the projector
    NUM_QUERIES = 128

    # OUTPUT_DIM is inferred automatically from the LLM hidden size
    NUM_HEADS = 8


    # ==================================================
    # Protein alignment parameters
    # ==================================================

    # Total protein prefix length fed into the LLM
    # = NUM_QUERIES + <protein_start> + <protein_end>
    # Default: 128 + 2 = 130
    PROTEIN_PREFIX_LEN = NUM_QUERIES + 2


    # ==================================================
    # Data loading parameters
    # ==================================================

    # Maximum length of the textual input (prompt + answer)
    # Used in dataloader
    MAX_TEXT_LEN = 8192


    # ==================================================
    # Default paths
    # ==================================================

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Qwen base model path
    DEFAULT_QWEN_PATH = "Qwen/Qwen3-8B-Base"

    # ESM model identifier or local cache directory
    # Used only for reference; ESM is typically loaded via `esm.pretrained`
    DEFAULT_ESM_MODEL = "esm2_t33_650M_UR50D"

    # Root dataset directory (protein-only datasets)
    DEFAULT_DATA_PATH = os.path.abspath(
        os.path.join(CURRENT_DIR, "Mol-Instructions/data/proteins/Protein-oriented_Instructions")
    )

    # Training output directory
    DEFAULT_OUTPUT_DIR = os.path.join(
        CURRENT_DIR, "qwen3_protein_sft_lora_results"
    )