#!/usr/bin/env bash
set -e

############################
# 基本路径配置
############################
BASE_DIR=$(pwd)

MODEL_OUTPUT_DIR="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval/Mol-Instructions/stage2_with_cot/mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage2_with_cot"
DATA_PATH="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/data/ChemLLMBench"          # 测试 / 验证数据目录
# merged_model_path="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval/Mol-Instructions/stage2_with_cot/mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage2_with_cot"

RESULT_DIR="${MODEL_OUTPUT_DIR}/inference_vllm"
mkdir -p "${RESULT_DIR}"

############################
# 模型权重
############################
LORA_PATH="${MODEL_OUTPUT_DIR}/lora_weights"
PROJECTOR_PATH="${MODEL_OUTPUT_DIR}/projector.pt"

############################
# 推理参数
############################
MAX_SEQ_LEN=8192
MAX_NEW_TOKENS=4096
TEMPERATURE=0.7
TOP_P=0.9
BATCH_SIZE=1   # 当前推理代码是 sample-level，保持 1
EMBED_BATCH_SIZE=64  # 嵌入层批处理大小，可以适当增大以提高效率

############################
# 分子投影器配置（⚠ 必须和训练一致）
############################
NUM_QUERIES=128
MOL_INPUT_DIM=768
MOL_NUM_HEADS=8

############################
# 并行配置
############################
NUM_PROCS=1          # >1 可多进程分片
PROC_INDEX=0         # 当前进程 id
GPU_ID=0             # 使用的 GPU

############################
# 结果路径
############################
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RESULT_PATH="${RESULT_DIR}/results_${TIMESTAMP}_p${PROC_INDEX}.json"

############################
# CUDA 环境
############################
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export TOKENIZERS_PARALLELISM=false

############################
# 运行推理
############################
python /zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/code_train_sft/train_sft_stage2.py \
  --mode vllm \
  --output_dir "${MODEL_OUTPUT_DIR}" \
  --data_path "${DATA_PATH}" \
  --test_data_path "${DATA_PATH}" \
  --lora_path "${LORA_PATH}" \
  --projector_path "${PROJECTOR_PATH}" \
  --max_seq_length ${MAX_SEQ_LEN} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --temperature ${TEMPERATURE} \
  --top_p ${TOP_P} \
  --num_queries ${NUM_QUERIES} \
  --mol_input_dim ${MOL_INPUT_DIM} \
  --mol_num_heads ${MOL_NUM_HEADS} \
  --proc_index ${PROC_INDEX} \
  --num_procs ${NUM_PROCS} \
  --inference_results_path "${RESULT_PATH}" \
  --embed_batch_size ${EMBED_BATCH_SIZE}

echo "✅ Inference finished. Results saved to:"
echo "   ${RESULT_PATH}"
