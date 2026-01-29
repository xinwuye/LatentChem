import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scienceplots
plt.style.use(['science','no-latex','ieee'])
# plt.rcParams["text.usetex"] = False
# plt.style.use('science')
rcParams['axes.prop_cycle'] = plt.cycler(color=[
    '#9eaad1',  # blue
    '#f59790',  # orange
    '#dacfe5',  # green
    '#f5dbb6',  # red
    '#cde2e8',
    '#c8d4e9'
    
])

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
    df.columns = [c.lower() for c in df.columns]

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
        if name=="drd":
            name="DRD2"
        elif name=="gsk":
            name="GSK3B"
        elif name=="jnk":   
            name="JNK3"
        elif name=="logp":
            name="LogP"
        elif name=="qed":
            name="QED"
        elif name=="solubility":
            name="Solubility"
        plt.errorbar(
            df["layer"],
            df["vs_answer_spearman"],
            yerr=df["vs_answer_p"],
            marker="o",
            capsize=3,
            linewidth=1,
            label=name,
        )

    plt.xlabel("Latent Thinking Step",fontsize=13)
    plt.ylabel("Output SMILES Structural Info. (Spearman Corr.)",fontsize=13)
    # plt.title("Output SMILES Structural Info. (Spearman Corr.)")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", ".svg"))
    plt.close()



def plot_input_smiles(all_results, out_path):
    """画 input_smiles（带 error bar）"""
    plt.figure(figsize=(8, 5))

    for name, df in all_results.items():

        if name=="drd":
            name="DRD-2"
        elif name=="gsk":
            name="GSK-3β"
        elif name=="jnk":   
            name="JNK"
        elif name=="logp":
            name="LogP"
        elif name=="qed":
            name="QED"
        elif name=="solubility":
            name="Solubility"
        plt.errorbar(
            df["layer"],
            df["vs_logp_spearman"],
            yerr=df["vs_logp_p"],
            marker="s",
            capsize=3,
            linewidth=1,
            label=name,
        )

    plt.xlabel("Latent Thinking Step",fontsize=13)
    plt.ylabel("Input SMILES Structural Info. (Spearman Corr.)",fontsize=13)
    # plt.title("Input SMILES Structural Info. (Spearman Corr.)")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", ".svg"))
    plt.close()



if __name__ == "__main__":
    ROOT_DIR = "/BioLatent/Bio-LatentCOT/matrix/result"

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