#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}/code_train_sft"
data_path="/public/home/xinwuye/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot"

echo "Starting latentchem14b stage 1..."

accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --qwen_size 14b \
  --training_stage 1 \
  --data_path "${data_path}" \
  --epochs_per_stage 3 \
  --output_dir ./outputs/stage1-latentchem14b \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0 \
  "$@"
