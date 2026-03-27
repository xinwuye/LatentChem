#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}/code_train_sft"

data_path="/public/home/xinwuye/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot"
output_dir="./outputs/baseline-textcot"

echo "Starting text-only stage2 SFT..."

accelerate launch --multi_gpu --num_processes 8 train_text.py \
  --data_path "${data_path}" \
  --training_stage 2 \
  --output_dir "${output_dir}" \
  --epochs 3 \
  --batch_size 8 \
  --grad_accum 1 \
  --lr 2e-4 \
  --max_seq_length 8192 \
  --bf16 true \
  --gradient_checkpointing true \
  "$@"
