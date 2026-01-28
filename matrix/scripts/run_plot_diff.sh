#!/bin/bash
# run_plot.sh
# 用法: ./run_plot.sh /path/to/top_folder

# 检查是否传入了参数
# if [ $# -lt 1 ]; then
#     echo "Usage: $0 <top_folder>"
#     exit 1
# fi

TOP_FOLDER="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/refined"
PYTHON_SCRIPT="/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/matrix/matrix_plot.py"  # Python脚本文件名

# 遍历所有子文件夹
find "$TOP_FOLDER" -type d | while read SUBFOLDER; do
    # 检查是否有 *_refined*.npy 文件
    FILES=$(find "$SUBFOLDER" -maxdepth 1 -name "*_refined*.npy")
    if [ -n "$FILES" ]; then
        echo "Processing folder: $SUBFOLDER"
        python3 "$PYTHON_SCRIPT" --folder "$SUBFOLDER"
    fi
done
