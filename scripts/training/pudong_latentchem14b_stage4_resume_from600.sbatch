#!/bin/bash
#SBATCH -J latentchem14b_stage4_resume_600
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

hf_base="/tmp/${USER}/latentchem14b_stage4_resume_600_${SLURM_JOB_ID}"
mkdir -p "${hf_base}/datasets"
export HF_HOME="${hf_base}"
export HF_DATASETS_CACHE="${hf_base}/datasets"

cd /public/home/xinwuye/Bio-LatentCOT
bash scripts/training/train_latentchem14b_stage4.sh \
  --resume_from_checkpoint ./outputs/stage4-latentchem14b/stage4/checkpoint-600
