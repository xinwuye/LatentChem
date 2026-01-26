#!/usr/bin/env bash
set -euo pipefail
#merge_json_folder.sh
# =========================
# 输入参数
# =========================
# =========================
model="nocot"
INPUT_FOLDER="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/outputs/noncot-chemllmbench_07/results"
OUTPUT_FILE="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/outputs/nocot-chemllmbench_07/results/merged.json"
mode="score"
# =========================
# 运行 Python 脚本
# =========================
PYTHON_BIN="python"  # 如果你用 conda/venv，可改成对应的 python

# 切换到 Python 脚本所在目录
cd /zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval/ || exit 1

# 检查 OUTPUT_FILE 是否已存在
if [[ -f "${OUTPUT_FILE}" ]]; then
    echo "Output file already exists: ${OUTPUT_FILE}"
    echo "Skipping merge process."
else
    echo "Merging JSON files in folder: ${INPUT_FOLDER}"
    echo "Output will be saved to: ${OUTPUT_FILE}"

    ${PYTHON_BIN} data_merge.py \
        --input_folder "${INPUT_FOLDER}" \
        --output_file "${OUTPUT_FILE}"

    echo "Done."
fi

# 切换到 Python 脚本所在目录
cd /zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval/ || exit 1

# 可选：激活 Python 虚拟环境
# source /zengdaojian/zhangjia/anaconda3/bin/activate biolatenecot_dev3

# 运行 Python 脚本
python group_results.py \
    --result_path ${OUTPUT_FILE} \
    --log_name nonlatent \
    --dataset_paths ../data/ChemLLMBench/chemllmbench


# 切换到脚本所在目录
cd /zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval || exit 1

# 可选：激活 Python 虚拟环境
# source /path/to/venv/bin/activate

# 运行 Python 脚本
python eval_results.py \
    --result_path  $OUTPUT_FILE \
    --log_name ${model} \
    --dataset_paths /zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/data/ChemLLMBench/chemllmbench \
    --num_samples 1 \
    --mode ${mode}
