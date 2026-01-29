#!/bin/bash

# Script to run eval_results.py with parameters
# Usage: ./run_eval_results.sh <result_path> <log_name> <dataset_paths> [num_samples] [mode]

RESULT_PATH="/BioLatent/Bio-LatentCOT/outputs/exp_chemllmbench_stage4_1_5_no/results/merge.json"
LOG_NAME="stage_4_inference_result_1_5_no"
DATASET_PATHS="/BioLatent/Bio-LatentCOT/data/ChemLLMBench/chemllmbench"
NUM_SAMPLES=1
MODE="record"
# result_dir="/BioLatent/Bio-LatentCOT/outputs/exp_chemllmbench_stage4_0_7_no/results/merge.json"

echo "Running eval_results.py with the following parameters:"
echo "  Result path: $RESULT_PATH"
echo "  Log name: $LOG_NAME"
echo "  Dataset path: $DATASET_PATHS"
echo "  Num samples: $NUM_SAMPLES"
echo "  Mode: $MODE"
echo "----------------------------------------"

# Change to the parent directory to run the script properly
cd 

# Run the Python script with the specified parameters
python BioLatent/Bio-LatentCOT/eval/eval_results.py \
    --result_path "$RESULT_PATH" \
    --log_name "$LOG_NAME" \
    --dataset_paths "$DATASET_PATHS" \
    --num_samples "$NUM_SAMPLES" \
    --mode "$MODE" \
    # --result_path "$result_dir"

# Check the exit status
EXIT_CODE=$?
echo "----------------------------------------"
echo "Script completed with exit code: $EXIT_CODE"

exit $EXIT_CODE
