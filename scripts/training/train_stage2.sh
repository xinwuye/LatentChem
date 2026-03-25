#!/bin/bash

set -euo pipefail

# Stage 2: Training with LoRA and Projector from Stage 1
echo "Starting Training Stage 2..."
cd code_train_sft 
accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 2 \
  --epochs_per_stage 3 \
  --lora_path ./outputs/stage1/stage1/lora_weights \
  --projector_path ./outputs/stage1/stage1/mm_projector.pt \
  --output_dir ./outputs/stage2 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0 \
  "$@"

cd ..
