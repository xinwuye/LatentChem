#!/bin/bash
#SBATCH -J bio_smoke_stage2
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
accelerate launch --multi_gpu --num_processes 2 train_stage3.py \
  --training_stage 2 \
  --data_path /public/home/xinwuye/Bio-LatentCOT/ChemCotDataset/chemcotbench-cot \
  --epochs_per_stage 1 \
  --output_dir ./outputs/debug-stage2 \
  --batch_size 1 \
  --grad_accum 1 \
  --lr 2e-4 \
  --max_seq_length 1024 \
  --cf_lambda 0.2 --cf_margin 0.1 \
  --cf_prob 1.0
'
