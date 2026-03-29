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

data_path="/public/home/xinwuye/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot"
stage3_output="./outputs/latentchem-seed${seed}-stage3/stage3"
stage4_output="./outputs/latentchem-seed${seed}-stage4-correct"
lora_path="${stage3_output}/lora_weights"
projector_path="${stage3_output}/mm_projector.pt"

if [[ ! -d "${lora_path}" ]]; then
  echo "Missing stage3 LoRA directory: ${lora_path}" >&2
  exit 1
fi
if [[ ! -f "${projector_path}" ]]; then
  echo "Missing stage3 projector checkpoint: ${projector_path}" >&2
  exit 1
fi
if [[ -e "${stage4_output}" ]]; then
  echo "Refusing to reuse existing output path: ${stage4_output}" >&2
  exit 1
fi

accelerate launch --multi_gpu --num_processes 8 train_grpo_try2.py \
  --seed "${seed}" \
  --run_name stage4 \
  --data_path "${data_path}" \
  --lora_path "${lora_path}" \
  --projector_path "${projector_path}" \
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
  --freeze_bio_updater true --freeze_bio_thinker true --freeze_task_thinker true
