#!/usr/bin/env bash
set -euo pipefail

# exp
EXP_NAME=<exp_name>
CKPT_DIR_NAME=<ckpt_name>
DATASET_NAME=ChemCoTBench
CUDA_DEVICES=0,1
QWEN_MODEL_NAME=<base_model_name>

# inference
BATCH_SIZE=1
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.8
SAMPLE_COUNT=1

# script path
SCRIPT_PATH="code_train_sft/train_sft_stage2.py"
MERGE_SCRIPT="code_train_sft/merge_and_extract_embed.py"

OUTPUT_DIR="outputs/${EXP_NAME}"
STAGE1_DIR="outputs/${CKPT_DIR_NAME}"
LORA_PATH="${STAGE1_DIR}/lora_weights"
PROJECTOR_PATH="${STAGE1_DIR}/projector.pt"
MERGED_DIR="${STAGE1_DIR}/merged"
DATA_PATH="data/${DATASET_NAME}"

PYTHON_BIN="${PYTHON_BIN:-python}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_NAME="${EXP_NAME}_${TIMESTAMP}"
INFERENCE_RESULTS_PATH="${OUTPUT_DIR}/inference_results_${TIMESTAMP}.json"

echo "========== Stage-2 Inference & Eval Runner =========="
echo "EXP_NAME:             ${EXP_NAME}"
echo "CKPT_DIR_NAME:        ${CKPT_DIR_NAME}"
echo "DATASET_NAME:         ${DATASET_NAME}"
echo "SCRIPT_PATH:          ${SCRIPT_PATH}"
echo "MERGE_SCRIPT:         ${MERGE_SCRIPT}"
echo "OUTPUT_DIR:           ${OUTPUT_DIR}"
echo "STAGE1_DIR:           ${STAGE1_DIR}"
echo "LORA_PATH:            ${LORA_PATH}"
echo "PROJECTOR_PATH:       ${PROJECTOR_PATH}"
echo "MERGED_DIR:           ${MERGED_DIR}"
echo "DATA_PATH:            ${DATA_PATH}"
echo "INFERENCE_RESULTS:    ${INFERENCE_RESULTS_PATH}"
echo "LOG_NAME (for eval):  ${LOG_NAME}"
if [[ -n "${CUDA_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
fi
echo "PYTHON:               ${PYTHON_BIN}"
echo "QWEN_MODEL_NAME:      ${QWEN_MODEL_NAME}"
echo "====================================================="

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: inference script not found: ${SCRIPT_PATH}"
  exit 3
fi

if [[ ! -f "${MERGE_SCRIPT}" ]]; then
  echo "ERROR: merge script not found: ${MERGE_SCRIPT}"
  exit 3
fi

if [[ ! -d "${LORA_PATH}" ]]; then
  echo "ERROR: Stage-1 LoRA dir not found: ${LORA_PATH}"
  exit 4
fi

if [[ ! -f "${PROJECTOR_PATH}" ]]; then
  echo "ERROR: Stage-1 projector ckpt not found: ${PROJECTOR_PATH}"
  exit 5
fi

if [[ ! -e "${DATA_PATH}" ]]; then
  echo "ERROR: dataset not found: ${DATA_PATH}"
  exit 6
fi

if [[ -z "${QWEN_MODEL_NAME}" || "${QWEN_MODEL_NAME}" == "<qwen_model_name>" ]]; then
  echo "ERROR: base model name not specified: ${QWEN_MODEL_NAME}"
  exit 7
fi

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/stage2_inference_and_eval_${TIMESTAMP}.log"

if [[ -n "${CUDA_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
  echo "Exported CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# merge
echo "$(date +'%Y-%m-%d %H:%M:%S') - Preparing merged model directory: ${MERGED_DIR}" | tee -a "${LOG_FILE}"

# skip merge if merged artifacts already exist
if [[ -f "${MERGED_DIR}/embeddings.safetensors" ]]; then
  echo "Found existing merged artifacts at ${MERGED_DIR}. Skipping merge step." | tee -a "${LOG_FILE}"
else
  mkdir -p "${MERGED_DIR}"
  MERGE_CMD=(
    "${PYTHON_BIN}" "${MERGE_SCRIPT}"
    --qwen_model_name "${QWEN_MODEL_NAME}"
    --lora_weights_path "${LORA_PATH}"
    --output_dir "${MERGED_DIR}"
  )

  echo "$(date +'%Y-%m-%d %H:%M:%S') - Launching merge_and_extract_embed..." | tee -a "${LOG_FILE}"
  echo "Running merge command:" | tee -a "${LOG_FILE}"
  printf " %s " "${MERGE_CMD[@]}" | tee -a "${LOG_FILE}"
  echo "" | tee -a "${LOG_FILE}"

  if "${MERGE_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "$(date +'%Y-%m-%d %H:%M:%S') - Merge finished successfully." | tee -a "${LOG_FILE}"
  else
    echo "$(date +'%Y-%m-%d %H:%M:%S') - ERROR: Merge failed. See log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    exit 8
  fi

  # after check
  if [[ ! -f "${MERGED_DIR}/embeddings.safetensors" ]]; then
    echo "ERROR: Merge finished but embeddings.safetensors not found in ${MERGED_DIR}" | tee -a "${LOG_FILE}"
    exit 9
  fi
fi

# inference
echo "$(date +'%Y-%m-%d %H:%M:%S') - Launching inference..." | tee -a "${LOG_FILE}"

INFER_CMD=(
  "${PYTHON_BIN}" "${SCRIPT_PATH}"
  --mode vllm
  --output_dir "${OUTPUT_DIR}"
  --data_path "${DATA_PATH}"
  --lora_path "${LORA_PATH}"
  --projector_path "${PROJECTOR_PATH}"
  --merged_model_path "${MERGED_DIR}"                
  --inference_results_path "${INFERENCE_RESULTS_PATH}"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --sample_count "${SAMPLE_COUNT}"
  --wandb_project "biolatentcot-stage2"   
  --wandb_run_name "${LOG_NAME}"
  --wandb_entity ""                       
)

echo "Running inference command:" | tee -a "${LOG_FILE}"
printf " %s " "${INFER_CMD[@]}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# run inference
if "${INFER_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"; then
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Inference finished successfully." | tee -a "${LOG_FILE}"
else
  echo "$(date +'%Y-%m-%d %H:%M:%S') - ERROR: Inference failed. See log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
  exit 10
fi

# check inference results existence
if [[ ! -f "${INFERENCE_RESULTS_PATH}" ]]; then
  echo "ERROR: inference results not found: ${INFERENCE_RESULTS_PATH}" | tee -a "${LOG_FILE}"
  exit 11
fi

# evaluation
if [[ ! -d "eval" ]]; then
  echo "ERROR: eval dir not found" | tee -a "${LOG_FILE}"
  exit 12
fi

EVAL_CMD=(
  "${PYTHON_BIN}" "eval/eval_results.py"
  --result_path "${RESULT_PATH}"
  --log_name "${LOG_NAME}"
)

echo "Running evaluation command:" | tee -a "../${LOG_FILE}"
printf " %s " "${EVAL_CMD[@]}" | tee -a "../${LOG_FILE}"
echo "" | tee -a "../${LOG_FILE}"

if "${EVAL_CMD[@]}" 2>&1 | tee -a "../${LOG_FILE}"; then
  echo "$(date +'%Y-%m-%d %H:%M:%S') - Evaluation finished successfully." | tee -a "../${LOG_FILE}"
else
  echo "$(date +'%Y-%m-%d %H:%M:%S') - ERROR: Evaluation failed. See log: ${LOG_FILE}" | tee -a "../${LOG_FILE}"
  popd > /dev/null
  exit 20
fi

popd > /dev/null

echo "All done."
echo "Inference results: ${INFERENCE_RESULTS_PATH}"
echo "Eval log name: ${LOG_NAME}"
echo "Full log: ${LOG_FILE}"
echo "====================================================="
