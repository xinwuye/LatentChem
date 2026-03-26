#!/bin/bash
#SBATCH -J bio_smoke_stage4
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

set +u
source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate latentchem_dev
set -u

cd /public/home/xinwuye/Bio-LatentCOT/code_train_sft
unset CUDA_VISIBLE_DEVICES

python - <<'PY'
import os
import torch

print("HOSTNAME=", os.uname().nodename)
print("torch.cuda.is_available=", torch.cuda.is_available())
print("torch.cuda.device_count=", torch.cuda.device_count())
PY

srun --ntasks=1 --gres=gpu:2 bash -lc '
set -euo pipefail
accelerate launch --multi_gpu --num_processes 2 train_grpo_try2.py \
  --run_name stage4 \
  --data_path /public/home/xinwuye/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot \
  --output_dir ./outputs/debug-stage4 \
  --is_both_latent false \
  --is_taskthinker true \
  --is_bioupdater true \
  --use_reward_answer_tag true \
  --use_reward_answer_type_validity true \
  --use_reward_answer_correctness_bench true \
  --batch_size 1 \
  --grad_accum 1 \
  --lr 1e-5 \
  --epochs 1 \
  --max_prompt_length 512 \
  --max_completion_length 256 \
  --num_generations 2 \
  --num_iterations 1 \
  --beta 0.0 \
  --use_liger \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_gpu_memory_utilization 0.2 \
  --vllm_max_model_len 1024 \
  --temperature 1.5 \
  --gradient_checkpointing \
  --freeze_bio_updater true \
  --freeze_bio_thinker true \
  --freeze_task_thinker true
'
