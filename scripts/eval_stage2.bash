set -euo pipefail

# exp
EXP_NAME=<exp_name>
CKPT_DIR_NAME=<ckpt_name>
DATASET_NAME=ChemCoTBench
CUDA_DEVICES=0,1

# inference
BATCH_SIZE=1
MAX_NEW_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.8

SCRIPT_PATH="code_train_sft/train_sft_stage2.py"
OUTPUT_DIR="outputs/${EXP_NAME}"
CKPT_DIR="outputs/${CKPT_DIR_NAME}"
LORA_PATH="${CKPT_DIR}/lora_weights"
PROJECTOR_PATH="${CKPT_DIR}/projector.pt"
DATA_PATH="data/${DATASET_NAME}"

PYTHON_BIN="python"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_NAME="${EXP_NAME}_${TIMESTAMP}"
INFERENCE_RESULTS_PATH="${OUTPUT_DIR}/inference_results_${TIMESTAMP}.json"

echo "========== Stage-2 Inference & Eval Runner =========="
echo "EXP_NAME:             ${EXP_NAME}"
echo "CKPT_DIR_NAME:        ${CKPT_DIR_NAME}"
echo "DATASET_NAME:         ${DATASET_NAME}"
echo "SCRIPT_PATH:          ${SCRIPT_PATH}"
echo "OUTPUT_DIR:           ${OUTPUT_DIR}"
echo "CKPT_DIR:             ${CKPT_DIR}"
echo "LORA_PATH:            ${LORA_PATH}"
echo "PROJECTOR_PATH:       ${PROJECTOR_PATH}"
echo "DATA_PATH:            ${DATA_PATH}"
echo "INFERENCE_RESULTS:    ${INFERENCE_RESULTS_PATH}"
echo "LOG_NAME (for eval):  ${LOG_NAME}"
if [[ -n "${CUDA_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICES}"
fi
echo "PYTHON:               ${PYTHON_BIN}"
echo "====================================================="

# Basic checks
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: Inference script not found : ${SCRIPT_PATH}"
  exit 3
fi

if [[ ! -d "${LORA_PATH}" ]]; then
  echo "ERROR: LoRA ckpt dir not found: ${LORA_PATH}"
  exit 4
fi

if [[ ! -f "${PROJECTOR_PATH}" ]]; then
  echo "ERROR: Projector ckpt file not found: ${PROJECTOR_PATH}"
  exit 5
fi

if [[ ! -e "${DATA_PATH}" ]]; then
  echo "ERROR: Data dir not found: ${DATA_PATH}"
  exit 6
fi

# logging setup
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/logs/stage2_inference_and_eval_${TIMESTAMP}.log"

# multi gpu multi process inference
echo "$(date +'%Y-%m-%d %H:%M:%S') - Launching inference per-GPU..." | tee -a "${LOG_FILE}"

# parse CUDA_VISIBLE_DEVICES into array
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_DEVICES}"
NUM_PROCS=${#GPU_ARRAY[@]}

PIDS=()
for idx in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[idx]}"
  PROC_INDEX="${idx}"

  # per-process result & log file
  RESULTS_PATH_PROC="${OUTPUT_DIR}/results/inference_results_${TIMESTAMP}.proc${PROC_INDEX}.json"
  LOG_FILE_PROC="${OUTPUT_DIR}/logs/stage2_inference_and_eval_${TIMESTAMP}.proc${PROC_INDEX}.log"

  CMD=( "${PYTHON_BIN}" "${SCRIPT_PATH}"
        --mode inference
        --output_dir "${OUTPUT_DIR}"
        --data_path "${DATA_PATH}"
        --lora_path "${LORA_PATH}"
        --projector_path "${PROJECTOR_PATH}"
        --inference_results_path "${RESULTS_PATH_PROC}"
        --batch_size "${BATCH_SIZE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --temperature "${TEMPERATURE}"
        --top_p "${TOP_P}"
        --wandb_project "biolatentcot-stage2"
        --wandb_run_name "${LOG_NAME}.proc${PROC_INDEX}"
        --wandb_entity ""
        --proc_index "${PROC_INDEX}"
        --num_procs "${NUM_PROCS}"
        --gpu "${GPU_ID}"
  )

  echo "Launching proc ${PROC_INDEX} on GPU ${GPU_ID}: ${CMD_ENV[*]} ${CMD[*]}" | tee -a "${LOG_FILE}"
  ( env CUDA_VISIBLE_DEVICES="${GPU_ID}" HF_DATASETS_CACHE="${OUTPUT_DIR}/hf_cache_proc${PROC_INDEX}" "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE_PROC}" ) &

  PIDS+=($!)
done

# wait for all processes to finish
for p in "${PIDS[@]}"; do
  wait "${p}" || {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - ERROR: One of inference processes failed (pid=${p})." | tee -a "${LOG_FILE}"
    exit 10
  }
done

echo "$(date +'%Y-%m-%d %H:%M:%S') - All inference processes finished successfully." | tee -a "${LOG_FILE}"

# cd and eval
if [[ ! -d "eval" ]]; then
  echo "ERROR: eval dir not found." | tee -a "${LOG_FILE}"
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