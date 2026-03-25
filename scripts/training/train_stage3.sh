#!/bin/bash

set -euo pipefail

# Stage 3: Training with TaskThinker and BioUpdater
echo "Starting Training Stage 3..."
cd code_train_sft 
accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 3 \
  --is_coconut false \
  --is_both_latent false \
  --is_taskthinker true \
  --is_bioupdater true \
  --epochs_per_stage 3 \
  --lora_path ./outputs/stage2/stage2/lora_weights \
  --projector_path ./outputs/stage2/stage2/mm_projector.pt \
  --output_dir ./outputs/stage3 \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0 \
  --freeze_llm true --freeze_projector true \
  "$@"

cd ..
