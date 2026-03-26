#!/bin/bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <seed>" >&2
  exit 1
fi

seed="$1"
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "Seed must be an integer, got: $seed" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repo_root}/code_train_sft"

stage1_output="./outputs/latentchem-seed${seed}-stage1"
stage2_output="./outputs/latentchem-seed${seed}-stage2"
stage3_output="./outputs/latentchem-seed${seed}-stage3"
stage4_output="./outputs/latentchem-seed${seed}-stage4"

echo "Starting latentchem seeded pipeline for seed ${seed}..."

accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 1 \
  --seed "${seed}" \
  --epochs_per_stage 3 \
  --output_dir "${stage1_output}" \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0

accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 2 \
  --seed "${seed}" \
  --epochs_per_stage 3 \
  --lora_path "${stage1_output}/stage1/lora_weights" \
  --projector_path "${stage1_output}/stage1/mm_projector.pt" \
  --output_dir "${stage2_output}" \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0

accelerate launch --multi_gpu --num_processes 8 train_stage3.py \
  --training_stage 3 \
  --seed "${seed}" \
  --is_coconut false \
  --is_both_latent false \
  --is_taskthinker true \
  --is_bioupdater true \
  --epochs_per_stage 3 \
  --lora_path "${stage2_output}/stage2/lora_weights" \
  --projector_path "${stage2_output}/stage2/mm_projector.pt" \
  --output_dir "${stage3_output}" \
  --batch_size 4 \
  --grad_accum 1 \
  --lr 2e-4 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0 \
  --freeze_llm true --freeze_projector true

accelerate launch --multi_gpu --num_processes 8 train_grpo_try2.py \
  --seed "${seed}" \
  --run_name stage4 \
  --lora_path "${stage3_output}/stage3/lora_weights" \
  --projector_path "${stage3_output}/stage3/mm_projector.pt" \
  --output_dir "${stage4_output}" \
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
  --log_reward_trace true \
  --freeze_bio_updater true --freeze_bio_thinker true --freeze_task_thinker true
