#!/bin/bash

# Script to run eval_querywise.py with parameters
# Usage: ./run_eval_querywise.sh <root_dir> <model_A> <model_B>

if [ $# -ne 3 ]; then
    echo "Usage: $0 <root_dir> <model_A> <model_B>"
    echo "Example: $0 /path/to/results my_model_A my_model_B"
    echo ""
    echo "Parameters:"
    echo "  root_dir: Root directory containing CSV files for evaluation"
    echo "  model_A: Name of the first model to compare"
    echo "  model_B: Name of the second model to compare"
    exit 1
fi

ROOT_DIR="$1"
MODEL_A="$2"
MODEL_B="$3"

echo "Running eval_querywise.py with the following parameters:"
echo "  Root directory: $ROOT_DIR"
echo "  Model A: $MODEL_A"
echo "  Model B: $MODEL_B"
echo "----------------------------------------"

# Change to the Bio-LatentCOT directory to ensure proper imports
cd /zengdaojian/zhangjia/BioLatent/Bio-LatentCOT

# Run the Python script with the specified parameters
python eval/query_wise/eval_querywise.py \
    --root_dir "$ROOT_DIR" \
    --model_A "$MODEL_A" \
    --model_B "$MODEL_B"

# Check the exit status
EXIT_CODE=$?
echo "----------------------------------------"
echo "Script completed with exit code: $EXIT_CODE"

exit $EXIT_CODE
python /zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval/query_wise/eval_querywise.py --root_dir /zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval/records/molecule_design --model_A inference_result_0_7_test --model_B stage_4_inference_re
sult_0_7_no
