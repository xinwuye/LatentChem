# InstructMol Evaluation Pipeline for Bio-LatentCOT

This directory contains the complete evaluation pipeline for running Bio-LatentCOT on InstructMol benchmark datasets:
- **ChEBI-20**: Molecule description generation (Mol2Cap)
- **Molecule-Oriented Instructions**: Multiple molecular tasks including description generation and reaction prediction

## Overview

Bio-LatentCOT is trained on ChemCOTBench and supports:
- ✅ Molecule Description Generation (SMILES → Text)
- ✅ Chemical Reaction Analysis (Forward Reaction Prediction, Retrosynthesis, Reagent Prediction)
- ❌ Molecule Generation from Text (Text → SMILES) - NOT supported

This pipeline evaluates Bio-LatentCOT against the InstructMol benchmarks for the supported tasks.

## Directory Structure

```
eval/InstructMol/
├── README.md                                    # This file
├── __init__.py                                  # Module init
├── core/                                         # Core utilities
│   └── __init__.py
├── preprocess_chebi20.py                        # ChEBI-20 preprocessing
├── preprocess_molecule_instructions.py          # Molecule-Oriented Instructions preprocessing
├── eval_chebi20.py                              # ChEBI-20 evaluation script
├── eval_molecule_instructions.py                # Molecule-Oriented Instructions evaluation
└── run_instructmol_eval.py                      # Unified pipeline runner
```

## Quick Start

### 1. Prerequisites

Make sure you have:
- Bio-LatentCOT model checkpoint (LoRA weights)
- ChEBI-20 dataset downloaded
- Molecule-Oriented Instructions dataset downloaded
- Required Python packages (see requirements)

```bash
# Install required packages
pip install selfies transformers datasets huggingface_hub
```

### 2. Download Datasets

The datasets should already be in place at:
- ChEBI-20: `instructmol/ChEBI-20/test_with_smiles.csv`
- Molecule-Oriented Instructions: `instructmol/Molecule-oriented_Instructions/*.json`

If not, run:

```bash
# Download ChEBI-20-MM (includes SMILES)
cd instructmol/ChEBI-20
python3 -c "
from huggingface_hub import hf_hub_download
import shutil

file_path = hf_hub_download(
    repo_id='liupf/ChEBI-20-MM',
    filename='test.csv',
    repo_type='dataset'
)
shutil.copy(file_path, 'test_with_smiles.csv')
"

# Molecule-Oriented Instructions should already be downloaded
# If not, download from: https://huggingface.co/datasets/zjunlp/Mol-Instructions
```

### 3. Run Complete Evaluation Pipeline

#### Option A: Evaluate Both Datasets (Recommended)

```bash
python3 eval/InstructMol/run_instructmol_eval.py \
    --dataset both \
    --lora_path /mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage3-lr2e-4-cf_margin01/stage3/lora_weights \
    --batch_size 8 \
    --max_new_tokens 1300
```

#### Option B: Evaluate ChEBI-20 Only

```bash
python3 eval/InstructMol/run_instructmol_eval.py \
    --dataset chebi20 \
    --lora_path /mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage3-lr2e-4-cf_margin01/stage3/lora_weights \
    --batch_size 8 \
    --max_new_tokens 1300
```

#### Option C: Evaluate Molecule-Oriented Instructions Only

```bash
python3 eval/InstructMol/run_instructmol_eval.py \
    --dataset molecule_instructions \
    --lora_path /mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage3-lr2e-4-cf_margin01/stage3/lora_weights \
    --batch_size 8 \
    --max_new_tokens 512
```

### 4. Quick Test with Small Sample

For quick testing with a small subset:

```bash
python3 eval/InstructMol/run_instructmol_eval.py \
    --dataset both \
    --lora_path /mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage3-lr2e-4-cf_margin01/stage3/lora_weights \
    --max_test_samples 100 \
    --max_samples 1000 \
    --batch_size 4
```

## Pipeline Steps

The evaluation pipeline consists of three main steps:

### Step 1: Preprocessing

Converts raw datasets to Bio-LatentCOT evaluation format.

#### ChEBI-20 Preprocessing:
```bash
python3 eval/InstructMol/preprocess_chebi20.py \
    --csv_path instructmol/ChEBI-20/test_with_smiles.csv \
    --output_dir instructmol/ChEBI-20/processed \
    --split test
```

**Output:** `instructmol/ChEBI-20/processed/chebi20_test.json` (~3,297 samples)

#### Molecule-Oriented Instructions Preprocessing:
```bash
python3 eval/InstructMol/preprocess_molecule_instructions.py \
    --input_dir instructmol/Molecule-oriented_Instructions \
    --output_dir instructmol/Molecule-oriented_Instructions/processed
```

**Output:** Processed JSON files for each task:
- `molecular_description_generation.json` (~298K samples)
- `forward_reaction_prediction.json` (~125K samples)
- `retrosynthesis.json` (~130K samples)
- `reagent_prediction.json` (~125K samples)

**Note:** The pipeline automatically:
- Converts SELFIES to SMILES (required for Bio-LatentCOT)
- Filters out description_guided_molecule_design (not supported by Bio-LatentCOT)
- Formats prompts and answers consistently

### Step 2: Inference

Runs Bio-LatentCOT model inference on the preprocessed data.

```bash
python3 code_train_sft/inference.py \
    --data_path <processed_json_file> \
    --lora_path <path_to_lora_weights> \
    --inference_results_path <output_json> \
    --batch_size 8 \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9
```

**Output:** JSON file with model predictions for each sample.

### Step 3: Evaluation

Computes metrics on the model predictions.

#### ChEBI-20 Evaluation:
```bash
python3 eval/InstructMol/eval_chebi20.py \
    --results_path <inference_results.json> \
    --gt_path instructmol/ChEBI-20/processed/chebi20_test.json \
    --output_dir eval_results/instructmol
```

**Metrics:** BLEU-2, BLEU-4, METEOR, ROUGE-1, ROUGE-2, ROUGE-L

#### Molecule-Oriented Instructions Evaluation:
```bash
# Evaluate all tasks
python3 eval/InstructMol/eval_molecule_instructions.py \
    --mode all \
    --results_dir <results_directory> \
    --gt_dir instructmol/Molecule-oriented_Instructions/processed \
    --output_dir eval_results/instructmol

# Or evaluate a single task
python3 eval/InstructMol/eval_molecule_instructions.py \
    --mode single \
    --results_path <task_inference_results.json> \
    --gt_path <task_gt.json> \
    --task_name molecular_description_generation \
    --output_dir eval_results/instructmol
```

**Metrics:**
- For description generation: BLEU-2, BLEU-4, METEOR, ROUGE-1, ROUGE-2, ROUGE-L
- For reaction tasks: Exact Match, BLEU, Levenshtein, Validity, MACCS Similarity, Morgan Similarity, RDK Similarity

## Configuration Options

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset to evaluate (chebi20/molecule_instructions/both) | Required |
| `--lora_path` | Path to LoRA weights | Required |
| `--projector_path` | Path to projector weights | None (optional) |
| `--batch_size` | Inference batch size | 8 |
| `--max_new_tokens` | Max tokens to generate | 512 |
| `--temperature` | Generation temperature | 0.7 |
| `--top_p` | Top-p sampling parameter | 0.9 |
| `--max_test_samples` | Max samples for inference (testing) | None (all) |
| `--max_samples` | Max samples for preprocessing (testing) | None (all) |
| `--skip_preprocessing` | Skip preprocessing if files exist | False |
| `--skip_inference` | Skip inference if files exist | False |
| `--output_dir` | Directory for evaluation results | eval_results/instructmol |

### Path Configuration

You can override default paths:

```bash
python3 eval/InstructMol/run_instructmol_eval.py \
    --dataset both \
    --lora_path /path/to/lora \
    --chebi20_csv_path /path/to/chebi20.csv \
    --chebi20_processed_path /path/to/processed.json \
    --chebi20_results_path /path/to/results.json \
    --mol_instr_raw_dir /path/to/raw \
    --mol_instr_processed_dir /path/to/processed \
    --mol_instr_results_dir /path/to/results \
    --output_dir /path/to/eval_results
```

## Output Files

After running the pipeline, you'll find:

```
eval_results/instructmol/
├── eval_chebi20_results.json                    # ChEBI-20 evaluation results
├── eval_molecular_description_generation_results.json
├── eval_forward_reaction_prediction_results.json
├── eval_retrosynthesis_results.json
├── eval_reagent_prediction_results.json
└── eval_all_tasks_summary.json                  # Summary of all tasks
```

Each results file contains:
- Dataset name and task description
- Number of samples evaluated
- Computed metrics with scores

Example:
```json
{
  "dataset": "ChEBI-20",
  "task": "Molecule Description Generation",
  "num_samples": 3297,
  "metrics": {
    "bleu-2": 0.4521,
    "bleu-4": 0.3142,
    "meteor": 0.5823,
    "rouge-1": 0.6134,
    "rouge-2": 0.4234,
    "rouge-L": 0.5912
  }
}
```

## Troubleshooting

### Common Issues

1. **SELFIES decoding errors:**
   - Some SELFIES strings may fail to convert to SMILES
   - These samples are automatically skipped
   - Check preprocessing logs for conversion statistics

2. **Memory issues:**
   - Reduce `--batch_size` (try 4 or 2)
   - Use `--max_test_samples` to test with smaller subsets
   - Consider processing tasks separately

3. **Missing ground truth:**
   - Ensure preprocessing step completed successfully
   - Check that sample IDs align between inference results and ground truth

4. **Model checkpoint not found:**
   - Verify the `--lora_path` points to the correct LoRA weights directory
   - Default path: `/mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage3-lr2e-4-cf_margin01/stage3/lora_weights`

### Debugging

Enable verbose logging:
```bash
export PYTHONUNBUFFERED=1
python3 eval/InstructMol/run_instructmol_eval.py ... 2>&1 | tee eval_log.txt
```

## Supported Tasks

### ✅ Supported by Bio-LatentCOT

1. **Molecular Description Generation** (Mol2Cap)
   - Input: SMILES string
   - Output: Natural language description
   - Dataset: ChEBI-20, Molecule-Oriented Instructions

2. **Forward Reaction Prediction**
   - Input: Reactants + Reagents (SMILES)
   - Output: Products (SMILES)
   - Dataset: Molecule-Oriented Instructions

3. **Retrosynthesis**
   - Input: Product (SMILES)
   - Output: Reactants (SMILES)
   - Dataset: Molecule-Oriented Instructions

4. **Reagent Prediction**
   - Input: Reactants + Products (SMILES)
   - Output: Reagents (SMILES)
   - Dataset: Molecule-Oriented Instructions

### ❌ Not Supported by Bio-LatentCOT

1. **Description-Guided Molecule Design** (Cap2Mol)
   - Input: Natural language description
   - Output: SMILES string
   - Reason: Bio-LatentCOT generates text, not molecules

## Comparison with InstructMol

This evaluation pipeline is designed to produce results directly comparable to those reported in the InstructMol paper:

- Uses the same datasets (ChEBI-20, Molecule-Oriented Instructions)
- Applies the same evaluation metrics
- Follows the same train/test splits
- Reports metrics in the same format

### Key Differences

1. **Model Architecture:**
   - InstructMol: Multi-modal model with graph encoders
   - Bio-LatentCOT: SMILES-only model with latent Chain-of-Thought

2. **Input Format:**
   - InstructMol: Accepts molecular graphs
   - Bio-LatentCOT: Accepts SMILES strings only

3. **Training Data:**
   - InstructMol: Trained on Mol-Instructions dataset
   - Bio-LatentCOT: Trained on ChemCOTBench

## References

- **InstructMol Paper:** [InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery](https://arxiv.org/abs/2311.16208)
- **ChEBI-20 Dataset:** [Papers with Code](https://paperswithcode.com/dataset/chebi-20)
- **ChEBI-20-MM Dataset:** [Hugging Face](https://huggingface.co/datasets/liupf/ChEBI-20-MM)
- **Molecule-Oriented Instructions:** [Hugging Face](https://huggingface.co/datasets/zjunlp/Mol-Instructions)
- **Bio-LatentCOT:** (ChemCOTBench paper)

## Citation

If you use this evaluation pipeline, please cite:

```bibtex
@article{instructmol2023,
  title={InstructMol: Multi-Modal Integration for Building a Versatile and Reliable Molecular Assistant in Drug Discovery},
  author={...},
  journal={arXiv preprint arXiv:2311.16208},
  year={2023}
}

@article{biolatentcot2025,
  title={Bio-LatentCOT: ...},
  author={...},
  journal={...},
  year={2025}
}
```

## Contact

For issues or questions about this evaluation pipeline, please open an issue in the Bio-LatentCOT repository.
