#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

QWEN_SIZE="8b"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qwen_size)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --qwen_size" >&2
        exit 1
      fi
      QWEN_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

mapfile -t QWEN_INFO < <(python - "${QWEN_SIZE}" <<'PY'
import sys
from code_train_sft.config import ModelConfig

size = ModelConfig.normalize_qwen_size(sys.argv[1])
print(ModelConfig.get_qwen_hf_repo(size))
print(ModelConfig.get_qwen_path(size))
print(ModelConfig.get_qwen_display_name(size))
PY
)

if [[ "${#QWEN_INFO[@]}" -ne 3 ]]; then
  echo "Failed to resolve Qwen model info for size: ${QWEN_SIZE}" >&2
  exit 1
fi

QWEN_HF_REPO="${QWEN_INFO[0]}"
QWEN_LOCAL_DIR="${QWEN_INFO[1]}"
QWEN_DISPLAY_NAME="${QWEN_INFO[2]}"

# Download base models
echo "Downloading ${QWEN_DISPLAY_NAME}..."
huggingface-cli download --resume-download "${QWEN_HF_REPO}" --local-dir "${QWEN_LOCAL_DIR}"

echo "Downloading smi-ted..."
huggingface-cli download --resume-download ibm-research/materials.smi-ted --local-dir "${REPO_ROOT}/models/smi-ted"
