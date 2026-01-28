set -euo pipefail

# =========================
# exp
# =========================

EXP_NAME="124-taskthinker-bioupdater-logp"
CKPT_DIR="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/models/124-taskthinker-bioupdater/stage4"
DATASET_NAME=ChemCoTBench/chemcotbench
TEMPERATURE=1.5
IS_BOTH_LATENT=false
IS_BIOTHINKER=false
IS_TASKTHINKER=true
IS_BIOUPDATER=true
IS_BIOTHINKER_MULTI=false
TASKTHINKER_TYPE="mlp"
IS_TASKTHINKER_MULTI=false
IS_BIOUPDATER_MULTI=false
IS_BIOTHINKER_GATING=false
IS_TASKTHINKER_GATING=false
IS_BIOUPDATER_GATING=false
MASK_TASK_LATENT_SETPS=0
MASK_TASK_LATENT_NOISE_STD=1.0
SHUFFLE_TASK_LATENTS=false
MASK_TASK_LATENT_IMPLEMENTATION="noise"
TASK_LATENT_MAX_STEPS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-name)
      EXP_NAME="$2"
      shift 2
      ;;
    --exp_name)
      EXP_NAME="$2"
      shift 2
      ;;
    --ckpt-dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    --ckpt_dir)
      CKPT_DIR="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --is-both-latent)
      IS_BOTH_LATENT="$2"
      shift 2
      ;;
    --is_both_latent)
      IS_BOTH_LATENT="$2"
      shift 2
      ;;
    --task-latent-max-steps)
      TASK_LATENT_MAX_STEPS="$2"
      shift 2
      ;;
    --task_latent_max_steps)
      TASK_LATENT_MAX_STEPS="$2"
      shift 2
      ;;
    --is-biothinker)
      IS_BIOTHINKER="$2"
      shift 2
      ;;
    --is_biothinker)
      IS_BIOTHINKER="$2"
      shift 2
      ;;
    --is-taskthinker)
      IS_TASKTHINKER="$2"
      shift 2
      ;;
    --is_taskthinker)
      IS_TASKTHINKER="$2"
      shift 2
      ;;
    --is-bioupdater)
      IS_BIOUPDATER="$2"
      shift 2
      ;;
    --is_bioupdater)
      IS_BIOUPDATER="$2"
      shift 2
      ;;
    --is-biothinker-multi)
      IS_BIOTHINKER_MULTI="$2"
      shift 2
      ;;
    --is_biothinker_multi)
      IS_BIOTHINKER_MULTI="$2"
      shift 2
      ;;
    --is-taskthinker-multi)
      IS_TASKTHINKER_MULTI="$2"
      shift 2
      ;;
    --is_taskthinker_multi)
      IS_TASKTHINKER_MULTI="$2"
      shift 2
      ;;
    --is-bioupdater-multi)
      IS_BIOUPDATER_MULTI="$2"
      shift 2
      ;;
    --is_bioupdater_multi)
      IS_BIOUPDATER_MULTI="$2"
      shift 2
      ;;
    --is-biothinker-gating)
      IS_BIOTHINKER_GATING="$2"
      shift 2
      ;;
    --is_biothinker_gating)
      IS_BIOTHINKER_GATING="$2"
      shift 2
      ;;
    --is-taskthinker-gating)
      IS_TASKTHINKER_GATING="$2"
      shift 2
      ;;
    --is_taskthinker_gating)
      IS_TASKTHINKER_GATING="$2"
      shift 2
      ;;
    --is-bioupdater-gating)
      IS_BIOUPDATER_GATING="$2"
      shift 2
      ;;
    --is_bioupdater_gating)
      IS_BIOUPDATER_GATING="$2"
      shift 2
      ;;
    --mask-task-latent-steps)
      MASK_TASK_LATENT_SETPS="$2"
      shift 2
      ;;
    --mask_task_latent_steps)
      MASK_TASK_LATENT_SETPS="$2"
      shift 2
      ;;
    --mask-task-latent-noise-std)
      MASK_TASK_LATENT_NOISE_STD="$2"
      shift 2
      ;;
    --mask_task_latent_noise_std)
      MASK_TASK_LATENT_NOISE_STD="$2"
      shift 2
      ;;
    --shuffle-task-latents)
      SHUFFLE_TASK_LATENTS="$2"
      shift 2
      ;;
    --shuffle_task_latents)
      SHUFFLE_TASK_LATENTS="$2"
      shift 2
      ;;
    --mask-task-latent-implementation)
      MASK_TASK_LATENT_IMPLEMENTATION="$2"
      shift 2
      ;;
    --mask_task_latent_implementation)
      MASK_TASK_LATENT_IMPLEMENTATION="$2"
      shift 2
      ;;
    --taskthinker-type)
      TASKTHINKER_TYPE="$2"
      shift 2
      ;;
    --taskthinker_type)
      TASKTHINKER_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${EXP_NAME}" || -z "${CKPT_DIR}" ]]; then
  echo "Usage: $0 --exp-name <EXP_NAME> --ckpt-dir <CKPT_DIR> [...]"
  exit 1
fi


INCLUDE_TASKS=""
CUDA_DEVICES=1

# =========================
# inference config
# =========================
BATCH_SIZE=16
NUM_RETURN_SEQUENCES=1
MAX_NEW_TOKENS=2048
TOP_P=0.9
MAX_SEQ_LENGTH=8192

# Stage-3 specific
TRAINING_STAGE=3
C_THOUGHT=2
BIO_LATENT_LAMBDA=0.0
BIO_LATENT_ALPHA=0.5
MAX_COT_STRING_LEN=2048
MAX_TEST_SAMPLES=""

# =========================
# path
# =========================
SCRIPT_PATH="code_train_sft/inference.py"
OUTPUT_DIR="outputs/${EXP_NAME}"
LORA_PATH="${CKPT_DIR}/lora_weights"
PROJECTOR_PATH="${CKPT_DIR}/mm_projector.pt"
DATA_PATH="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/data/ChemCoTBench/chemcotbench"

PYTHON_BIN="python"

is_true() {
  local v="${1:-}"
  v="${v,,}"
  case "${v}" in
    1|true|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

is_positive() {
  local v="${1:-}"
  if [[ "${v}" =~ ^[1-9][0-9]*$ ]]; then
    return 0
  else
    return 1
  fi
}

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

LOG_PARTS=()
LOG_PARTS+=("${EXP_NAME}")

if is_true "${IS_BOTH_LATENT}"; then
  LOG_PARTS+=("BOTH_LATENT")
fi
if is_true "${IS_BIOTHINKER}"; then
  LOG_PARTS+=("BIOTHINKER")
fi
if is_true "${IS_TASKTHINKER}"; then
  LOG_PARTS+=("TASKTHINKER")
fi
if is_true "${IS_BIOUPDATER}"; then
  LOG_PARTS+=("BIOUPDATER")
fi
if is_true "${IS_BIOTHINKER_MULTI}"; then
  LOG_PARTS+=("BIOTHINKER_MULTI")
fi
if is_true "${IS_TASKTHINKER_MULTI}"; then
  LOG_PARTS+=("TASKTHINKER_MULTI")
fi
if is_true "${IS_BIOUPDATER_MULTI}"; then
  LOG_PARTS+=("BIOUPDATER_MULTI")
fi
if is_true "${IS_BIOTHINKER_GATING}"; then
  LOG_PARTS+=("BIOTHINKER_GATING")
fi
if is_true "${IS_TASKTHINKER_GATING}"; then
  LOG_PARTS+=("TASKTHINKER_GATING")
fi
if is_true "${IS_BIOUPDATER_GATING}"; then
  LOG_PARTS+=("BIOUPDATER_GATING")
fi
if is_positive "${MASK_TASK_LATENT_SETPS}"; then
  LOG_PARTS+=("MASK${MASK_TASK_LATENT_SETPS}")
  LOG_PARTS+="${MASK_TASK_LATENT_IMPLEMENTATION}"
fi
if is_true "${SHUFFLE_TASK_LATENTS}"; then
  LOG_PARTS+=("SHUFFLE_TASK_LATENTS")
fi
if [ "$TASKTHINKER_TYPE" = "identity" ]; then
  LOG_PARTS+=("IDENTITY")
fi

LOG_PARTS+=("T${TEMPERATURE}")
LOG_PARTS+=("${DATASET_NAME}")
LOG_PARTS+=("TASKMAX${TASK_LATENT_MAX_STEPS}")
LOG_PARTS+=("${TIMESTAMP}")

IFS='_'
RAW_LOG_NAME="${LOG_PARTS[*]}"
unset IFS
LOG_NAME="${RAW_LOG_NAME//\//_}"
LOG_NAME="${LOG_NAME//./}"
LOG_NAME="${LOG_NAME// /_}"

INFERENCE_RESULTS_PATH="${OUTPUT_DIR}/results/inference_results_${TIMESTAMP}.json"

echo "========== Stage-3 Inference Runner =========="
echo "EXP_NAME:                  ${EXP_NAME}"
echo "DATASET_NAME:              ${DATASET_NAME}"
echo "SCRIPT_PATH:               ${SCRIPT_PATH}"
echo "CKPT_DIR:                  ${CKPT_DIR}"
echo "LORA_PATH:                 ${LORA_PATH}"
echo "PROJECTOR_PATH:            ${PROJECTOR_PATH}"
echo "DATA_PATH:                 ${DATA_PATH}"
echo "TRAINING_STAGE:            ${TRAINING_STAGE}"
echo "C_THOUGHT:                 ${C_THOUGHT}"
echo "IS_BOTH_LATENT:            ${IS_BOTH_LATENT}"
echo "IS_BIOTHINKER:             ${IS_BIOTHINKER}"
echo "IS_TASKTHINKER:            ${IS_TASKTHINKER}"
echo "IS_BIOUPDATER:             ${IS_BIOUPDATER}"
echo "IS_BIOTHINKER_MULTI:       ${IS_BIOTHINKER_MULTI}"
echo "TASKTHINKER_TYPE:          ${TASKTHINKER_TYPE}"
echo "IS_TASKTHINKER_MULTI:      ${IS_TASKTHINKER_MULTI}"
echo "IS_BIOUPDATER_MULTI:       ${IS_BIOUPDATER_MULTI}"
echo "IS_BIOTHINKER_GATING:      ${IS_BIOTHINKER_GATING}"
echo "IS_TASKTHINKER_GATING:     ${IS_TASKTHINKER_GATING}"
echo "IS_BIOUPDATER_GATING:      ${IS_BIOUPDATER_GATING}"
echo "TASK_LATENT_MAX_STEPS:     ${TASK_LATENT_MAX_STEPS}"
echo "MASK_TASK_LATENT_SETPS:    ${MASK_TASK_LATENT_SETPS}"
echo "MASK_TASK_LATENT_IMPLEMENTATION: ${MASK_TASK_LATENT_IMPLEMENTATION}"
echo "SHUFFLE_TASK_LATENTS:      ${SHUFFLE_TASK_LATENTS}"
echo "MASK_TASK_LATENT_NOISE_STD: ${MASK_TASK_LATENT_NOISE_STD}"
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
    --is_biothinker "${IS_BIOTHINKER}"
    --is_taskthinker "${IS_TASKTHINKER}"
    --is_bioupdater "${IS_BIOUPDATER}"
    --is_biothinker_multi "${IS_BIOTHINKER_MULTI}"
    --taskthinker_type "${TASKTHINKER_TYPE}"
    --is_taskthinker_multi "${IS_TASKTHINKER_MULTI}"
    --is_bioupdater_multi "${IS_BIOUPDATER_MULTI}"
    --is_biothinker_gating "${IS_BIOTHINKER_GATING}"
    --is_taskthinker_gating "${IS_TASKTHINKER_GATING}"
    --is_bioupdater_gating "${IS_BIOUPDATER_GATING}"
    --mask_task_latent_steps "${MASK_TASK_LATENT_SETPS}"
    --mask_task_latent_implementation "${MASK_TASK_LATENT_IMPLEMENTATION}"
    --shuffle_task_latents "${SHUFFLE_TASK_LATENTS}"
    --mask_task_latent_noise_std "${MASK_TASK_LATENT_NOISE_STD}"
    --bio_latent_lambda "${BIO_LATENT_LAMBDA}"
    --bio_latent_alpha "${BIO_LATENT_ALPHA}"
    --max_cot_string_len "${MAX_COT_STRING_LEN}"
    --task_latent_max_steps "${TASK_LATENT_MAX_STEPS}"
    --inference_results_path "${RESULTS_PATH_PROC}"
    --proc_index "${PROC_INDEX}"
    --num_procs "${NUM_PROCS}"
    --gpu "${GPU_ID}"
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
  "${PYTHON_BIN}" "eval_results.py"
  --result_path "../${INFERENCE_RESULTS_PATH}"
  --log_name "${LOG_NAME}"
  --dataset_paths "../${DATA_PATH}"
  --num_samples "${NUM_RETURN_SEQUENCES}"
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

echo "All done."
echo "Inference results: ${INFERENCE_RESULTS_PATH}"
echo "Eval log name: ${LOG_NAME}"
echo "Full log: ${LOG_FILE}"
echo "====================================================="
