set -euo pipefail

# =========================
# exp
# =========================
EXP_NAME=exp_chemllmbench_stage3_1_5_no
CKPT_DIR_NAME=chemllmbench_stage3_1_5_no
DATASET_NAME=ChemLLMBench/chemllmbench
INCLUDE_TASKS=""
CUDA_DEVICES=0,1,2,3,4

# =========================
# inference config
# =========================
BATCH_SIZE=8
NUM_RETURN_SEQUENCES=1
MAX_NEW_TOKENS=2048
TEMPERATURE=1.5
TOP_P=0.9
MAX_SEQ_LENGTH=8192

# Stage-3 specific
TRAINING_STAGE=3
C_THOUGHT=2
IS_BOTH_LATENT=false
BIO_LATENT_LAMBDA=0.0
BIO_LATENT_ALPHA=0.5
MAX_COT_STRING_LEN=2048
TASK_LATENT_MAX_STEPS=10
MAX_TEST_SAMPLES=""
TENSOR_PARALLEL_SIZE=5

# =========================
# path
# =========================
SCRIPT_PATH="code_train_sft/inference.py"
OUTPUT_DIR="outputs/${EXP_NAME}"
CKPT_DIR="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/models/final_model/stage4-lr2e-4-cf_margin01-freeze_llm-freeze_projector-1247-lr1e-5-temp15-nonlatent/stage4"
LORA_PATH="${CKPT_DIR}/lora_weights"
PROJECTOR_PATH="${CKPT_DIR}/mm_projector.pt"
DATA_PATH="data/${DATASET_NAME}"

PYTHON_BIN="python"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_NAME="${EXP_NAME}_${TIMESTAMP}"
LOG_NAME="${LOG_NAME//\//_}"
INFERENCE_RESULTS_PATH="${OUTPUT_DIR}/results/inference_results_${TIMESTAMP}.json"

echo "========== Stage-3 Inference Runner =========="
echo "EXP_NAME:                  ${EXP_NAME}"
echo "CKPT_DIR_NAME:             ${CKPT_DIR_NAME}"
echo "DATASET_NAME:              ${DATASET_NAME}"
echo "SCRIPT_PATH:               ${SCRIPT_PATH}"
echo "CKPT_DIR:                  ${CKPT_DIR}"
echo "LORA_PATH:                 ${LORA_PATH}"
echo "PROJECTOR_PATH:            ${PROJECTOR_PATH}"
echo "DATA_PATH:                 ${DATA_PATH}"
echo "TRAINING_STAGE:            ${TRAINING_STAGE}"
echo "C_THOUGHT:                 ${C_THOUGHT}"
echo "IS_BOTH_LATENT:            ${IS_BOTH_LATENT}"
echo "INFERENCE_RESULTS_PATH:    ${INFERENCE_RESULTS_PATH}"
echo "LOG_NAME:                  ${LOG_NAME}"
if [[ -n "${CUDA_DEVICES}" ]]; then
  echo "CUDA_VISIBLE_DEVICES:      ${CUDA_DEVICES}"
fi
echo "PYTHON:                    ${PYTHON_BIN}"
echo "============================================="

# =========================
# Basic checks
# =========================
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

# =========================
# logging
# =========================
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/results"
LOG_FILE="${OUTPUT_DIR}/logs/stage3_inference_${TIMESTAMP}.log"

echo "$(date +'%Y-%m-%d %H:%M:%S') - Launching stage-3 inference..." | tee -a "${LOG_FILE}"

# =========================
# multi-gpu multi-process
# =========================
IFS=',' read -ra GPU_ARRAY <<< "${CUDA_DEVICES}"
NUM_PROCS=${#GPU_ARRAY[@]}

PIDS=()
for idx in "${!GPU_ARRAY[@]}"; do
  GPU_ID="${GPU_ARRAY[idx]}"
  PROC_INDEX="${idx}"

  RESULTS_PATH_PROC="${OUTPUT_DIR}/results/inference_results_${TIMESTAMP}.proc${PROC_INDEX}.json"
  LOG_FILE_PROC="${OUTPUT_DIR}/logs/stage3_inference_${TIMESTAMP}.proc${PROC_INDEX}.log"

  CMD=(
    "${PYTHON_BIN}" "${SCRIPT_PATH}"
    --data_path "${DATA_PATH}"
    --include_tasks ${INCLUDE_TASKS}
    --lora_path "${LORA_PATH}"
    --projector_path "${PROJECTOR_PATH}"
    --training_stage "${TRAINING_STAGE}"
    --c_thought "${C_THOUGHT}"
    --batch_size "${BATCH_SIZE}"
    --num_return_sequences "${NUM_RETURN_SEQUENCES}"
    --max_seq_length "${MAX_SEQ_LENGTH}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --is_both_latent "${IS_BOTH_LATENT}"
    --bio_latent_lambda "${BIO_LATENT_LAMBDA}"
    --bio_latent_alpha "${BIO_LATENT_ALPHA}"
    --max_cot_string_len "${MAX_COT_STRING_LEN}"
    --task_latent_max_steps "${TASK_LATENT_MAX_STEPS}"
    --inference_results_path "${RESULTS_PATH_PROC}"
    --proc_index "${PROC_INDEX}"
    --num_procs "${NUM_PROCS}"
    --gpu "${GPU_ID}"
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  )

  if [[ -n "${MAX_TEST_SAMPLES}" ]]; then
    CMD+=( --max_test_samples "${MAX_TEST_SAMPLES}" )
  fi

  echo "Launching proc ${PROC_INDEX} on GPU ${GPU_ID}" | tee -a "${LOG_FILE}"
  echo "CMD: ${CMD[*]}" | tee -a "${LOG_FILE}"

  (
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        HF_DATASETS_CACHE="${OUTPUT_DIR}/hf_cache_proc${PROC_INDEX}" \
        "${CMD[@]}" \
        2>&1 | tee -a "${LOG_FILE_PROC}"
  ) &

  PIDS+=($!)
done

# =========================
# wait
# =========================
for p in "${PIDS[@]}"; do
  wait "${p}" || {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - ERROR: One of inference processes failed (pid=${p})." | tee -a "${LOG_FILE}"
    exit 10
  }
done

echo "$(date +'%Y-%m-%d %H:%M:%S') - All stage-3 inference processes finished successfully." | tee -a "${LOG_FILE}"

# cd and eval
if [[ ! -d "eval" ]]; then
  echo "ERROR: eval dir not found." | tee -a "${LOG_FILE}"
  exit 12
fi

cd eval || {
  echo "ERROR: Failed to cd to eval dir." | tee -a "${LOG_FILE}"
  exit 14
}

EVAL_CMD=(
  "${PYTHON_BIN}" "results_merge.py"
  --result_path "../${INFERENCE_RESULTS_PATH}"
  --log_name "${LOG_NAME}"
  --dataset_paths "../${DATA_PATH}"
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
echo "====================================================="d
