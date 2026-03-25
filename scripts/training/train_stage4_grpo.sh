#!/bin/bash

set -euo pipefail

# Stage 4: GRPO Training
echo "Starting Training Stage 4 (GRPO)..."
cd code_train_sft 
accelerate launch --multi_gpu --num_processes 8 train_grpo_try2.py \
  --run_name stage4 \
  --lora_path ./outputs/stage3/stage3/lora_weights \
  --projector_path ./outputs/stage3/stage3/mm_projector.pt \
  --output_dir ./outputs/stage4 \
  --is_both_latent false \
  --is_taskthinker true \
  --is_bioupdater true \
  --use_reward_answer_tag true \
  --use_reward_answer_type_validity true \
  --use_reward_answer_correctness_bench true \
  --batch_size 16 \
  --grad_accum 1 \
  --lr 1e-5 \
  --epochs 1 \
  --max_prompt_length 2048 \
  --max_completion_length 2048 \
  --num_generations 8 \
  --num_iterations 1 \
  --beta 0.0 \
  --use_liger \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.3 \
  --vllm_max_model_len 4096 \
  --temperature 1.5 \
  --gradient_checkpointing \
  --freeze_bio_updater true --freeze_task_thinker true \
  "$@"

cd ..
