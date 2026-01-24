import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import argparse

def load_and_flatten_csv(filepath):
    """加载 CSV 并展平为一维数值数组（跳过非数值）"""
    try:
        df = pd.read_csv(filepath, header=None)
    except Exception as e:
        print(f"⚠️ 无法读取 {filepath}: {e}")
        return None

    # 展平为一维
    flat = df.values.flatten()
    # 只保留数值类型（过滤 NaN、inf、非数字）
    numeric = pd.to_numeric(flat, errors='coerce')
    clean = numeric[~np.isnan(numeric)]
    return clean

def main(folder_path, output_file="spearman_results.txt",task="logp"):
    # 定义两个参考文件
    ref1 = f"molecular_similarity_matrix_{task}_answer.csv"
    ref2 = f"molecular_similarity_matrix_{task}.csv"

    ref1_path = os.path.join(folder_path, ref1)
    ref2_path = os.path.join(folder_path, ref2)

    if not os.path.exists(ref1_path):
        raise FileNotFoundError(f"参考文件未找到: {ref1_path}")
    if not os.path.exists(ref2_path):
        raise FileNotFoundError(f"参考文件未找到: {ref2_path}")

    # 加载参考向量
    ref1_vec = load_and_flatten_csv(ref1_path)
    ref2_vec = load_and_flatten_csv(ref2_path)

    if ref1_vec is None or ref2_vec is None:
        raise ValueError("无法加载参考文件")

    # 获取所有其他 CSV 文件
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    target_files = [f for f in all_files if f not in (ref1, ref2)]

    if not target_files:
        print("⚠️ 没有找到待比较的 CSV 文件。")
        return

    # 准备输出
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write("File\tvs_answer_spearman\tp_value\tvs_lop_spearman\tp_value\n")
        print(f"正在处理 {len(target_files)} 个文件...")

        for filename in sorted(target_files):
            filepath = os.path.join(folder_path, filename)
            test_vec = load_and_flatten_csv(filepath)

            if test_vec is None:
                line = f"{filename}\tERROR\t-\tERROR\t-\n"
                f_out.write(line)
                print(f"❌ 跳过 {filename}：加载失败")
                continue

            # 确保长度一致（取最小长度，或报错）
            min_len = min(len(ref1_vec), len(test_vec))
            if min_len < 2:
                line = f"{filename}\tTOO_SHORT\t-\tTOO_SHORT\t-\n"
                f_out.write(line)
                print(f"❌ 跳过 {filename}：数据太短")
                continue

            # 截断到相同长度（保守做法）
            r1 = ref1_vec[:min_len]
            t1 = test_vec[:min_len]
            r2 = ref2_vec[:min_len]
            t2 = test_vec[:min_len]

            try:
                corr1, p1 = spearmanr(r1, t1)
                corr2, p2 = spearmanr(r2, t2)
                line = f"{filename}\t{corr1:.6f}\t{p1:.6e}\t{corr2:.6f}\t{p2:.6e}\n"
                f_out.write(line)
                print(f"✅ {filename}: answer={corr1:.4f} (p={p1:.2e}), lop={corr2:.4f} (p={p2:.2e})")
            except Exception as e:
                line = f"{filename}\tCOMPUTE_ERROR\t-\tCOMPUTE_ERROR\t-\n"
                f_out.write(line)
                print(f"❌ 计算失败 {filename}: {e}")

    print(f"\n🎉 结果已保存至: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算文件夹中 CSV 与两个参考文件的 Spearman 相关性")
    parser.add_argument("folder", help="包含 CSV 文件的文件夹路径")
    parser.add_argument("-o", "--output", default="spearman_results.txt", help="输出 TXT 文件名")
    parser.add_argument("-t", "--task", default="qed", help="任务类型")
    args = parser.parse_args()

    main(args.folder, args.output, args.task)