#!/bin/bash

# 用法: ./run_eval.sh /path/to/root_dir modelA modelB

# 参数
ROOT_DIR="/BioLatent/Bio-LatentCOT/eval/records"
MODEL_A="nocot"
MODEL_B="nonlatent"

# 检查参数
if [ -z "$ROOT_DIR" ] || [ -z "$MODEL_A" ] || [ -z "$MODEL_B" ]; then
    echo "Usage: $0 ROOT_DIR MODEL_A MODEL_B"
    exit 1
fi

# 遍历根目录下的所有子文件夹
for task_dir in "$ROOT_DIR"/*/; do
    if [ -d "$task_dir" ]; then
        echo "Evaluating task in folder: $task_dir"
        python3 /BioLatent/Bio-LatentCOT/eval/querywise/eval_querywise.py --root_dir "$task_dir" --model_A "$MODEL_A" --model_B "$MODEL_B"
    fi
done
