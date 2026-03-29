#!/bin/bash
#SBATCH -J baseline_text_grpo_lr1e6
#SBATCH --partition=gpu
#SBATCH --exclude=server13,server16
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=5-00:00:00
#SBATCH -o slurm_logs/%x-%j.out
#SBATCH -e slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

set +u
source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate latentchem_dev
set -u

hf_base="/tmp/${USER}/baseline_text_grpo_lr1e6_${SLURM_JOB_ID}"
mkdir -p "${hf_base}/datasets"
export HF_HOME="${hf_base}"
export HF_DATASETS_CACHE="${hf_base}/datasets"

cd /public/home/xinwuye/Bio-LatentCOT
unset CUDA_VISIBLE_DEVICES

bash scripts/training/train_baseline_text_grpo.sh \
  --lr 1e-6 \
  --output_dir ./outputs/baseline-text-grpo-correct
