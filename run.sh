export CONDA_PATH=/mnt/afs/L202500070/miniconda3 
echo "source $CONDA_PATH/etc/profile.d/conda.sh" >> /root/.bashrc
echo "source $CONDA_PATH/etc/profile.d/conda.sh" >> /root/.profile
source /mnt/afs/L202500070/miniconda3/etc/profile.d/conda.sh
conda activate biolatenecot_dev
cd /mnt/afs/L202500070/yimeng/Bio-LatentCOT/code_train_sft
accelerate launch --multi_gpu --num_processes 8 train_sft_stage2.py \
  --mode train \
  --include_cot false \
  --output_dir ./outputs/stage1_no_cot \
  --batch_size 3 \
  --grad_accum 1 \
  --epochs 3 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lr 5e-5 \
  --finetuned_layers all-linear