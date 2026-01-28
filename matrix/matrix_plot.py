import os
import glob
import re
import argparse
import numpy as np
import matplotlib
import math
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matrix_utils import norm_1, norm_inf, norm_2, norm_fro, norm_nuclear


# ---------------- argparse ----------------
parser = argparse.ArgumentParser(
    description="Compute norms of begin → refined matrix differences and plot results"
)
parser.add_argument(
    "--folder", type=str, required=True,
    help="Path to folder containing *_begin*.npy and *_refined*.npy files"
)
args = parser.parse_args()
folder_path = args.folder
# ------------------------------------------


def leading_int(fname):
    """Extract leading integer from filename for matching begin/refined"""
    base = os.path.basename(fname)
    m = re.match(r"(\d+)", base)
    return int(m.group(1)) if m else None


# ---------- 搜索文件 ----------
refined_files = glob.glob(os.path.join(folder_path, "*_refined*.npy"))
begin_files = glob.glob(os.path.join(folder_path, "*_begin*.npy"))

refined_files = sorted(set(refined_files), key=leading_int)
begin_files = sorted(set(begin_files), key=leading_int)

print(f"Found {len(refined_files)} refined files")
print(f"Found {len(begin_files)} begin files")


# ---------- 建立 begin 映射 ----------
begin_map = {}
for f in begin_files:
    key = leading_int(f)
    if key is not None:
        begin_map[key] = f


# ---------- 范数函数 ----------
norm_funcs = {
    "norm1": norm_1,
    "norm_inf": norm_inf,
    "norm2": norm_2,
    "fro": norm_fro,
    "nuclear": norm_nuclear,
}

per_norm_per_file = {k: [] for k in norm_funcs}
skipped = []


# ---------- 主循环 ----------
for f in refined_files:
    try:
        key = leading_int(f)
        if key not in begin_map:
            skipped.append((f, "no matching begin file"))
            continue

        # ---------- load refined ----------
        refined_arr = np.load(f, allow_pickle=True)
        refined_elements = list(refined_arr)

        if len(refined_elements) < 1:
            skipped.append((f, "empty refined file"))
            continue

        # ---------- load begin ----------
        begin_arr = np.load(begin_map[key], allow_pickle=True)

        # 兼容 begin.npy 的不同格式
        if isinstance(begin_arr, np.ndarray) and begin_arr.ndim == 2:
            begin_mat = begin_arr
        elif isinstance(begin_arr, (list, np.ndarray)):
            begin_mat = np.asarray(begin_arr[0])
        else:
            skipped.append((f, "unsupported begin format"))
            continue

        # ---------- 拼接 begin + refined ----------
        elements = [[begin_mat]] + refined_elements

        print(
            f"Processing {os.path.basename(f)} | "
            f"begin shape = {elements[0][0].shape}, "
            f"steps = {len(elements)}"
        )

        cur_norms = {k: [] for k in norm_funcs}

        for t in range(len(elements) - 1):
            try:
                A = np.asarray(elements[t][0])
                B = np.asarray(elements[t + 1][0])
            except Exception:
                for k in norm_funcs:
                    cur_norms[k].append(np.nan)
                continue

            if A.shape != B.shape:
                for k in norm_funcs:
                    cur_norms[k].append(np.nan)
                continue

            diff = B - A
            # ---------- 处理 NaN/Inf ----------
            diff = np.nan_to_num(diff, nan=0.0, posinf=1e10, neginf=-1e10)

            for name, func in norm_funcs.items():
                try:
                    val = func(diff)
                    if not np.isscalar(val):
                        val = np.mean(val)
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0  # 用 0 填充，保证绘图从 step 0 开始
                    cur_norms[name].append(float(val))
                except Exception:
                    cur_norms[name].append(0.0)

        for name in norm_funcs:
            per_norm_per_file[name].append(cur_norms[name])

    except Exception as e:
        skipped.append((f, str(e)))


print("Processed:", {k: len(v) for k, v in per_norm_per_file.items()})
if skipped:
    print("Skipped examples:", skipped[:5])


# ---------- 输出目录 ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(
    script_dir, "plot", "figure", os.path.basename(folder_path)
)
os.makedirs(out_dir, exist_ok=True)


# ---------- 聚合 & 绘图 ----------
for name, per_files in per_norm_per_file.items():
    if not per_files:
        continue

    max_len = max(len(x) for x in per_files)
    pos_vals = [[] for _ in range(max_len)]

    for norms in per_files:
        for i, v in enumerate(norms):
            pos_vals[i].append(v)  # 即使 v=0 也加入

    avg_vals = np.array([
       math.log(np.mean(v)) if len(v) > 0 and np.mean(v) > 0 else 0.0
        for v in pos_vals
    ])

    x = np.arange(len(avg_vals))  # step 0 = begin → refined[0]

    plt.figure(figsize=(7, 4))
    plt.plot(x, avg_vals, marker="o")
    plt.xlabel("Step (0 = begin → refined[0])")
    plt.ylabel(f"Average {name}")
    plt.title(f"Average {name} across files")
    plt.grid(True)

    out_path = os.path.join(out_dir, f"{name}_avg_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[{name}] saved → {out_path}")


print("All done. Results in:", out_dir)
