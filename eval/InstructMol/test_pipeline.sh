#!/bin/bash

# Quick test script for InstructMol evaluation pipeline
# This script tests the pipeline with a small subset of data

set -e  # Exit on error

echo "=============================================================================="
echo "InstructMol Evaluation Pipeline - Quick Test"
echo "=============================================================================="

# Configuration
LORA_PATH="/mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage3-lr2e-4-cf_margin01/stage3/lora_weights"
MAX_SAMPLES=100
BATCH_SIZE=4

echo ""
echo "Test Configuration:"
echo "  LoRA Path: $LORA_PATH"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Check if LoRA weights exist
if [ ! -d "$LORA_PATH" ]; then
    echo "ERROR: LoRA weights not found at: $LORA_PATH"
    echo "Please update the LORA_PATH variable in this script."
    exit 1
fi

# Test 1: Preprocess ChEBI-20 (small sample)
echo "=============================================================================="
echo "Test 1: ChEBI-20 Preprocessing"
echo "=============================================================================="

python3 eval/InstructMol/preprocess_chebi20.py \
    --csv_path instructmol/ChEBI-20/test_with_smiles.csv \
    --output_dir instructmol/ChEBI-20/test_processed \
    --split test

echo "✓ ChEBI-20 preprocessing completed"

# Test 2: Preprocess Molecule-Oriented Instructions (small sample)
echo ""
echo "=============================================================================="
echo "Test 2: Molecule-Oriented Instructions Preprocessing"
echo "=============================================================================="

python3 eval/InstructMol/preprocess_molecule_instructions.py \
    --input_dir instructmol/Molecule-oriented_Instructions \
    --output_dir instructmol/Molecule-oriented_Instructions/test_processed \
    --max_samples $MAX_SAMPLES

echo "✓ Molecule-Oriented Instructions preprocessing completed"

# Test 3: ChEBI-20 Inference (small sample)
echo ""
echo "=============================================================================="
echo "Test 3: ChEBI-20 Inference (${MAX_SAMPLES} samples)"
echo "=============================================================================="

python3 code_train_sft/inference.py \
    --data_path instructmol/ChEBI-20/test_processed/chebi20_test.json \
    --lora_path $LORA_PATH \
    --inference_results_path outputs/instructmol/test_chebi20_inference.json \
    --batch_size $BATCH_SIZE \
    --max_new_tokens 256 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_test_samples $MAX_SAMPLES

echo "✓ ChEBI-20 inference completed"

# Test 4: ChEBI-20 Evaluation
echo ""
echo "=============================================================================="
echo "Test 4: ChEBI-20 Evaluation"
echo "=============================================================================="

python3 eval/InstructMol/eval_chebi20.py \
    --results_path outputs/instructmol/test_chebi20_inference.json \
    --gt_path instructmol/ChEBI-20/test_processed/chebi20_test.json \
    --output_dir eval_results/instructmol/test

echo "✓ ChEBI-20 evaluation completed"

# Test 5: Molecule-Oriented Instructions Inference (molecular_description_generation only)
echo ""
echo "=============================================================================="
echo "Test 5: Molecule-Oriented Instructions Inference"
echo "=============================================================================="

python3 code_train_sft/inference.py \
    --data_path instructmol/Molecule-oriented_Instructions/test_processed/molecular_description_generation.json \
    --lora_path $LORA_PATH \
    --inference_results_path outputs/instructmol/test_molecular_description_generation_inference.json \
    --batch_size $BATCH_SIZE \
    --max_new_tokens 256 \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_test_samples $MAX_SAMPLES

echo "✓ Molecule-Oriented Instructions inference completed"

# Test 6: Molecule-Oriented Instructions Evaluation
echo ""
echo "=============================================================================="
echo "Test 6: Molecule-Oriented Instructions Evaluation"
echo "=============================================================================="

python3 eval/InstructMol/eval_molecule_instructions.py \
    --mode single \
    --results_path outputs/instructmol/test_molecular_description_generation_inference.json \
    --gt_path instructmol/Molecule-oriented_Instructions/test_processed/molecular_description_generation.json \
    --task_name molecular_description_generation \
    --output_dir eval_results/instructmol/test

echo "✓ Molecule-Oriented Instructions evaluation completed"

# Summary
echo ""
echo "=============================================================================="
echo "Pipeline Test Summary"
echo "=============================================================================="
echo "✓ All tests passed successfully!"
echo ""
echo "Test results saved to: eval_results/instructmol/test/"
echo ""
echo "To run the full evaluation pipeline, use:"
echo "  python3 eval/InstructMol/run_instructmol_eval.py --dataset both --lora_path $LORA_PATH"
echo ""
echo "=============================================================================="
