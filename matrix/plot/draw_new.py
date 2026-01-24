import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def parse_result_txt(path):
    """
    读取 result.txt，返回按 layer 排序的 DataFrame
    列: layer, vs_answer_spearman, vs_answer_p,
        vs_logp_spearman, vs_logp_p
    """
    try:
        df = pd.read_csv(path, sep=r'\s+', engine='python', comment='#', header=0)
    except Exception as e:
        print(f"  ⚠ pandas read failed for {path}: {e}")
        return pd.DataFrame()

    if df.shape[1] < 5:
        print(f"  ⚠ file has <5 cols: {path}")
        return pd.DataFrame()

    cols = df.columns.tolist()
    file_col = cols[0]
    ans_corr_col = cols[1]
    ans_p_col = cols[2]
    lop_corr_col = cols[3]
    lop_p_col = cols[4]

    rows = []
    for _, r in df.iterrows():
        file_name = str(r[file_col])
        m = re.search(r"layer(\d+)", file_name)
        if not m:
            continue

        try:
            rows.append({
                "layer": int(m.group(1)),
                "vs_answer_spearman": float(r[ans_corr_col]),
                "vs_answer_p": float(r[ans_p_col]),
                "vs_logp_spearman": float(r[lop_corr_col]),
                "vs_logp_p": float(r[lop_p_col]),
            })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def collect_all_results(root_dir):
    """
    遍历所有 results.txt，返回：
    {
        name1: DataFrame,
        name2: DataFrame,
        ...
    }
    """
    all_results = {}

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith("results.txt"):
                continue

            path = os.path.join(root, fname)
            print(f"Processing: {path}")

            df = parse_result_txt(path)
            if df.empty:
                print("  ⚠ no valid layer rows found.")
                continue

            # 用文件名或父目录名区分曲线
            key = os.path.splitext(fname)[0].split('_')[1]
            all_results[key] = df

    return all_results


def plot_output_smiles(all_results, out_path):
    """画 output_smiles（带 error bar）"""
    plt.figure(figsize=(8, 5))

    for name, df in all_results.items():
        plt.errorbar(
            df["layer"],
            df["vs_answer_spearman"],
            yerr=df["vs_answer_p"],
            marker="o",
            capsize=3,
            linewidth=1,
            label=name,
        )

    plt.xlabel("Layer")
    plt.ylabel("Spearman Correlation")
    plt.title("Output SMILES Similarity vs Layer")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



def plot_input_smiles(all_results, out_path):
    """画 input_smiles（带 error bar）"""
    plt.figure(figsize=(8, 5))

    for name, df in all_results.items():
        plt.errorbar(
            df["layer"],
            df["vs_logp_spearman"],
            yerr=df["vs_logp_p"],
            marker="s",
            capsize=3,
            linewidth=1,
            label=name,
        )

    plt.xlabel("Layer")
    plt.ylabel("Spearman Correlation")
    plt.title("Input SMILES Similarity vs Layer")
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()



if __name__ == "__main__":
    ROOT_DIR = "/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/matrix/result"

    all_results = collect_all_results(ROOT_DIR)
    if not all_results:
        raise RuntimeError("No valid results.txt found.")

    plot_output_smiles(
        all_results,
        os.path.join(ROOT_DIR, "all_output_smiles_vs_layer.png")
    )

    plot_input_smiles(
        all_results,
        os.path.join(ROOT_DIR, "all_input_smiles_vs_layer.png")
    )

    print("✅ Done. Two figures saved.")
