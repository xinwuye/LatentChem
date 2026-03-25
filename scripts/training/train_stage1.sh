#!/bin/bash

set -euo pipefail

# Stage 1: Initial Training
echo "Starting Training Stage 1..."
cd code_train_sft 
accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 1 \
  --epochs_per_stage 3 \
  --output_dir ./outputs/stage1 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0 \
  "$@"

cd ..
