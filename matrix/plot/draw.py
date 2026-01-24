import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_result_txt(path):
    """
    读取 result.txt，返回按 layer 排序的 DataFrame，列:
    layer, vs_answer_spearman, vs_answer_p, vs_logp_spearman, vs_logp_p
    只保留文件名中含 layer(\d+) 的行，refined 自动忽略。
    """
    # 尝试使用 pandas 解析任意空白分隔表格（支持 tab/空格）
    try:
        df = pd.read_csv(path, sep=r'\s+', engine='python', comment='#', header=0)
    except Exception as e:
        print(f"  ⚠ pandas read failed for {path}: {e}")
        return pd.DataFrame()

    # 必须至少有 5 列（File + 4 数值列）
    if df.shape[1] < 5:
        print(f"  ⚠ file has <5 cols: {path}")
        return pd.DataFrame()

    # 约定列映射（优先按 header 名称，若不明确则按位置）
    cols = df.columns.tolist()
    file_col = cols[0]
    # 假设列顺序： File, ans_corr, ans_p, lop_corr, lop_p
    ans_corr_col = cols[1]
    ans_p_col = cols[2]
    lop_corr_col = cols[3]
    lop_p_col = cols[4]

    rows = []
    for _, r in df.iterrows():
        file_name = str(r[file_col])
        m = re.search(r"layer(\d+)", file_name)
        if not m:
            # 不含 layer 的行忽略 (包括 refined)
            continue
        layer = int(m.group(1))
        try:
            ans_corr = float(r[ans_corr_col])
            ans_p = float(r[ans_p_col])
            lop_corr = float(r[lop_corr_col])
            lop_p = float(r[lop_p_col])
        except Exception:
            # 数值转换失败则跳过该行
            continue

        rows.append({
            "layer": layer,
            "vs_answer_spearman": ans_corr,
            "vs_answer_p": ans_p,
            "vs_logp_spearman": lop_corr,
            "vs_logp_p": lop_p,
        })

    if not rows:
        return pd.DataFrame()

    out_df = pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
    return out_df


def plot_and_save(df, out_path, title):
    """
    画图并保存（两条折线 + error bar）
    """
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        df["layer"],
        df["vs_answer_spearman"],
        yerr=df["vs_answer_p"],
        marker="o",
        capsize=4,
        label="output_smiles",
    )

    plt.errorbar(
        df["layer"],
        df["vs_logp_spearman"],
        yerr=df["vs_logp_p"],
        marker="s",
        capsize=4,
        label="input_smiles",
    )

    plt.xlabel("Layer")
    plt.ylabel("Spearman Correlation")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    plt.close()


def process_all_results(root_dir):
    """
    遍历 root_dir，处理所有以 result.txt 结尾的文件
    """
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith("results.txt"):
                continue

            result_path = os.path.join(root, fname)
            print(f"Processing: {result_path}")

            df = parse_result_txt(result_path)
            if df.empty:
                print("  ⚠ no valid layer rows found (refined ignored or parse failed).")
                continue

            # 输出文件名包含原文件名以区分
            base = os.path.splitext(fname)[0]
            out_fig = os.path.join(root, f"spearman_vs_layer_{base}.png")
            plot_and_save(df, out_fig, title=base.split('_')[1])
            print(f"  ✅ saved {out_fig}")


if __name__ == "__main__":
    # ← 改为你的根目录（脚本会递归查找所有 *result.txt）
    ROOT_DIR = "/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/matrix/result"
    process_all_results(ROOT_DIR)
