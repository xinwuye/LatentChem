
import os
import shutil
import re

# ===== 路径配置 =====
src_dir = "/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/refine_corrected/refined"  # 原始文件夹
dst_dirs = [
    "refined_drd",
    "refined_gsk",
    "refined_jnk",
    "refined_logp",
    "refined_qed",
    "refined_solubility",
]

# 创建目标文件夹
for d in dst_dirs:
    os.makedirs(d, exist_ok=True)

# 匹配文件开头数字
pattern = re.compile(r"^(\d+)")

# 遍历文件
for fname in os.listdir(src_dir):
    match = pattern.match(fname)
    if not match:
        continue  # 文件名不以数字开头就跳过

    idx = int(match.group(1))

    # 根据序号选择目标文件夹
    if 1 <= idx <= 100:
        group = 0
    elif 101 <= idx <= 200:
        group = 1
    elif 201 <= idx <= 300:
        group = 2
    elif 301 <= idx <= 400:
        group = 3
    elif 401 <= idx <= 500:
        group = 4
    elif 501 <= idx <= 600:
        group = 5
    else:
        continue  # 超过范围的序号忽略

    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dirs[group], fname)

    shutil.move(src_path, dst_path)
    # 如果想复制而不删除原文件，改成：
    # shutil.copy(src_path, dst_path)

print("✅ 文件按序号分配完成")
