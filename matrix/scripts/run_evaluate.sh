#!/bin/bash

# run_spearman.sh
# 用途：运行 Spearman 相关性分析及相关预处理
# 用法: ./run_spearman.sh <任务名称>

set -e  # 遇错即停

# === 定义全局变量 ===
TASK=${1:-"logp"}  # 默认任务类型为 "qed"
BASE_DIR="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT"
SCRIPTS_DIR="$BASE_DIR/matrix"
INPUT_FOLDER="$BASE_DIR/matrix/result/biotoken_${TASK}"
OUTPUT_FILE="$BASE_DIR/matrix/result/spearman_${TASK}_results.txt"
PYTHON_SCRIPT="caculate_spearman_batch.py"
ANSWER_FILE="$BASE_DIR/outputs/124-taskthinker-bioupdater-${TASK}/results/inference_results_*.json"
MATCHED_FILES=$(ls $ANSWER_FILE 2>/dev/null)
if [ -z "$MATCHED_FILES" ]; then
    echo "❌ 错误: 未找到匹配的答案文件: $ANSWER_FILE"
    exit 1
fi

# 如果有多个匹配文件，取第一个
ANSWER_FILE=$(echo $MATCHED_FILES | awk '{print $1}')
# === 检查输入参数 ===
if [ -z "$TASK" ]; then
    echo "❌ 错误: 请提供任务名称 (例如 'qed' 或 'logp')"
    exit 1
fi

# === 路径检查 ===
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "❌ 错误: 输入文件夹不存在: $INPUT_FOLDER"
    exit 1
fi

# 转为绝对路径（避免相对路径问题）
INPUT_FOLDER="$(realpath "$INPUT_FOLDER")"
OUTPUT_DIR="$(dirname "$OUTPUT_FILE")"
OUTPUT_FILE="$(realpath "$OUTPUT_FILE")"

# 创建输出目录（如果不存在）
mkdir -p "$OUTPUT_DIR"

# === 检查 Python 脚本 ===
if [ ! -f "$SCRIPTS_DIR/$PYTHON_SCRIPT" ]; then
    echo "❌ 错误: 未找到 Python 脚本 '$PYTHON_SCRIPT'（请放在同一目录）"
    exit 1
fi

# === 检查 Python 环境 ===
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未安装 python3"
    exit 1
fi

# === 步骤 1: 将输入分子转化为相似度矩阵 ===
echo "🚀 步骤 1: 将输入分子转化为相似度矩阵..."
JSON_FILE="$BASE_DIR/data/ChemCoTBench/chemcotbench/mol_opt/${TASK}.json"
SIM_MATRIX_FILE="$BASE_DIR/matrix/result/biotoken_${TASK}/molecular_similarity_matrix_${TASK}.csv"

python3 "$SCRIPTS_DIR/extract_molecules.py" \
    --json_file "$JSON_FILE" \
    --output_file "$SIM_MATRIX_FILE"

echo "✅ 完成！输入分子相似度矩阵已保存至: $SIM_MATRIX_FILE"

# === 步骤 2: 提取输出答案并转化为相似度矩阵 ===
echo "🚀 步骤 2: 提取输出答案并转化为相似度矩阵..."

ANSWER_SIM_MATRIX_FILE="$BASE_DIR/matrix/result/biotoken_${TASK}/molecular_similarity_matrix_${TASK}_answer.csv"

python3 "$SCRIPTS_DIR/answer_evaluate.py" \
    --json_file "$JSON_FILE" \
    --answer_file "$ANSWER_FILE" \
    --output_file "$ANSWER_SIM_MATRIX_FILE" \
    --reference_path_file "$JSON_FILE"

echo "✅ 完成！输出答案相似度矩阵已保存至: $ANSWER_SIM_MATRIX_FILE"

# === 步骤 3: 计算平均值并保存 ===
echo "🚀 步骤 3: 计算平均值并保存..."
LOGP_DIR="$BASE_DIR/refine_corrected/${TASK}"
SAVE_PATH="$BASE_DIR/matrix/result/biotoken_${TASK}"

python3 "$SCRIPTS_DIR/biotoken_matrix_mean.py" \
    --task "$TASK" \
    --logp_dir "$LOGP_DIR" \
    --json_file_path "$JSON_FILE" \
    --save_path "$SAVE_PATH"

echo "✅ 完成！平均值结果已保存至: $SAVE_PATH"

# === 步骤 4: 运行 1-10 层相似性矩阵 ===
echo "🚀 步骤 4: 运行 1-10 层相似性矩阵..."
REFINED_DIR="$BASE_DIR/refined/${TASK}"

python3 "$SCRIPTS_DIR/biotokn_martix.py" \
    --task "$TASK" \
    --refined_dir "$REFINED_DIR" \
    --json_file_path "$JSON_FILE" \
    --save_dir "$SAVE_PATH" \
    --num_layers 10 \
    --prefix "biotoken"

echo "✅ 完成！1-10 层相似性矩阵已保存至: $SAVE_PATH"

# === 步骤 5: 运行 Spearman 相关性分析 ===
echo "🚀 步骤 5: 运行 Spearman 相关性分析..."
echo "📁 输入文件夹: $INPUT_FOLDER"
echo "📄 输出文件:   $OUTPUT_FILE"
echo "----------------------------------------"

python3 "$SCRIPTS_DIR/$PYTHON_SCRIPT" "$INPUT_FOLDER" -o "$OUTPUT_FILE" -t "$TASK"

echo "✅ 完成！Spearman 相关性分析结果已保存至: $OUTPUT_FILE"