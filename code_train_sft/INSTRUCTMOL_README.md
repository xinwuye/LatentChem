# InstructMol Dataloader Documentation

This document describes the newly added dataloader functionality for the InstructMol Molecule-oriented_Instructions dataset.

## Overview

The InstructMol dataset has been integrated into the existing dataloader infrastructure in `dataloader.py`, with automatic detection support in `inference.py`.

## Dataset Structure

The InstructMol Molecule-oriented_Instructions dataset contains JSON files with the following structure:

```json
{
    "instruction": "Please provide the HOMO-LUMO gap of this molecule.",
    "input": "[O][=C][C][N][C][Branch1][Ring1][C][=O][C][=C][Ring1][#Branch1]",
    "output": 0.1913,
    "metadata": {
        "task": "property prediction",
        "split": "train"
    }
}
```

### Dataset Files

Located at: `/mnt/afs/L202500070/yimeng/Bio-LatentCOT/instructmol/Molecule-oriented_Instructions/`

- `description_guided_molecule_design.json`
- `forward_reaction_prediction.json`
- `molecular_description_generation.json`
- `property_prediction.json`
- `reagent_prediction.json`
- `retrosynthesis.json`

## New Functions

### 1. `extract_instructmol_fields(example, is_eval=False)`

Extracts and formats fields from InstructMol dataset entries.

**Parameters:**
- `example`: Dict containing instruction, input, output, and metadata
- `is_eval`: Boolean, if True, labels are set to None (for inference)

**Returns:**
```python
{
    "query": str,           # Formatted instruction with answer template
    "input_smiles": list,   # List of SMILES strings
    "label": str or None,   # "<answer> output </answer>" format
    "cot": None,           # InstructMol has no CoT
    "cot_steps": None,     # InstructMol has no CoT steps
    "task": str            # Task name from metadata
}
```

**Features:**
- Automatically splits multiple SMILES in input by '.'
- Adds answer formatting instruction to queries
- Wraps labels in `<answer>` tags for consistency
- Handles eval mode by setting labels to None

### 2. `load_instructmol_data(path, max_len, is_coconut, scheduled_stage, c_thought, eval_mode)`

Loads and tokenizes the InstructMol dataset.

**Parameters:**
- `path`: Path to directory containing InstructMol JSON files
- `max_len`: Maximum sequence length (default: ModelConfig.MAX_TEXT_LEN)
- `is_coconut`: Not applicable for InstructMol (no CoT)
- `scheduled_stage`: Not applicable for InstructMol (no CoT)
- `c_thought`: Not applicable for InstructMol (no CoT)
- `eval_mode`: If True, loads without labels for inference

**Returns:**
HuggingFace Dataset with fields:
- `input_ids`: Tokenized input
- `attention_mask`: Attention mask
- `labels`: Training labels (None if eval_mode=True)
- `smiles`: List of SMILES strings

## Integration with Inference

The `load_test_data()` function in `inference.py` has been updated to automatically detect InstructMol datasets:

```python
from inference import load_test_data

# Automatically detects InstructMol format based on path
dataset = load_test_data(
    "/mnt/afs/L202500070/yimeng/Bio-LatentCOT/instructmol/Molecule-oriented_Instructions"
)
```

**Detection Rules:**
1. If path contains "Molecule-oriented_Instructions" → Uses `load_instructmol_data()`
2. If path contains "instructmol" (case-insensitive) → Uses `load_instructmol_data()`
3. If path contains "ChemCoTBench" → Uses `load_data()` with ChemCot format
4. Otherwise → Defaults to `load_data()` with warning

## Usage Examples

### Example 1: Load for Inference

```python
from dataloader import load_instructmol_data

# Load InstructMol dataset for evaluation
dataset = load_instructmol_data(
    path="/mnt/afs/L202500070/yimeng/Bio-LatentCOT/instructmol/Molecule-oriented_Instructions",
    max_len=2048,
    eval_mode=True
)

print(f"Loaded {len(dataset)} samples")
sample = dataset[0]
print(f"Input IDs: {sample['input_ids']}")
print(f"SMILES: {sample['smiles']}")
```

### Example 2: Run Inference

```bash
cd /mnt/afs/L202500070/yimeng/Bio-LatentCOT/

huggingface-cli login 

huggingface-cli download blc-org/A-secret-model-repo stage3-0111-2.tar.gz --repo-type model --local-dir . --local-dir-use-symlinks False

tar -xf stage3-0111-2.tar.gz --no-same-owner

cd code_train_sft

python inference.py \
    --data_path /mnt/afs/L202500070/yimeng/Bio-LatentCOT/instructmol/Molecule-oriented_Instructions \
    --lora_path /mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage3-lr2e-4-cf_margin01/stage3/lora_weights \
    --projector_path path/to/projector/weights \
    --inference_results_path ./outputs/instructmol_results.json \
    --batch_size 32 \
    --max_new_tokens 2048 \
    --temperature 0.7 \
    --training_stage 3
```

### Example 3: Load for Training

```python
from dataloader import load_instructmol_data

# Load InstructMol dataset for training
dataset = load_instructmol_data(
    path="/mnt/afs/L202500070/yimeng/Bio-LatentCOT/instructmol/Molecule-oriented_Instructions",
    max_len=2048,
    eval_mode=False  # Include labels for training
)

# Training data includes labels
sample = dataset[0]
print(f"Has labels: {sample['labels'] is not None}")
```

## Key Differences from ChemCot Format

| Feature | ChemCot | InstructMol |
|---------|---------|-------------|
| CoT Support | Yes (struct_cot field) | No |
| Data Format | Complex JSON with meta field | Simple instruction-input-output |
| SMILES Location | In meta.molecule or reactants/products | In input field |
| Label Source | Priority: gt → reference → output | Direct from output field |
| Query Processing | Extensive regex processing | Simple instruction formatting |

## Notes

1. **No CoT Support**: InstructMol dataset doesn't contain Chain-of-Thought reasoning steps, so `is_coconut` parameter has no effect
2. **Answer Formatting**: All queries are automatically appended with answer formatting instructions
3. **SMILES Parsing**: Multiple SMILES in the input field are automatically split by '.'
4. **Task Tracking**: Task information from metadata is preserved for evaluation

## Testing

A test script is available at `test_instructmol_dataloader.py` to verify the implementation:

```bash
cd /mnt/afs/L202500070/yimeng/Bio-LatentCOT/code_train_sft
python3 test_instructmol_dataloader.py
```

The test verifies:
- Field extraction logic
- Dataset loading in both eval and training modes
- Integration with inference.py

## File Locations

- Main dataloader: [code_train_sft/dataloader.py](code_train_sft/dataloader.py)
- Inference integration: [code_train_sft/inference.py](code_train_sft/inference.py)
- Test script: [code_train_sft/test_instructmol_dataloader.py](code_train_sft/test_instructmol_dataloader.py)
- This documentation: [code_train_sft/INSTRUCTMOL_README.md](code_train_sft/INSTRUCTMOL_README.md)
