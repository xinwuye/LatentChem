import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def load_similarity_matrix(csv_path):
    """
    从 CSV 文件加载相似度矩阵。
    假设第一列是行索引，第一行是列名。
    """
    df = pd.read_csv(csv_path, index_col=0)
    return df

def spearman_upper_triangle(mat1, mat2):
    """
    计算两个方阵上三角部分（不含对角线）的 Spearman 相关系数。
    """
    # 检查是否为方阵且形状相同
    if mat1.shape[0] != mat1.shape[1]:
        raise ValueError("Matrix 1 is not square.")
    if mat2.shape[0] != mat2.shape[1]:
        raise ValueError("Matrix 2 is not square.")
    if mat1.shape != mat2.shape:
        raise ValueError("Matrices must have the same shape.")
    
    # 获取上三角索引（k=1 表示跳过对角线）
    iu = np.triu_indices_from(mat1, k=1)
    vec1 = mat1[iu]
    vec2 = mat2[iu]
    
    # 计算 Spearman 相关
    rho, pval = spearmanr(vec1, vec2)
    return rho, pval, len(vec1)

# -----------------------------
# 主程序：替换你的文件路径
# -----------------------------
if __name__ == "__main__":
    # 替换为你的实际文件路径
    for i in range(0,1):
        csv_file_1 = "/BioLatent/Bio-LatentCOT/eval/silmilarity_matrix/molecular_similarity_matrix_lop_answer.csv"   # 例如 RDKit 指纹相似度
        csv_file_2 = "/BioLatent/Bio-LatentCOT/eval/silmilarity_matrix/qwen3_smiles_similarity_string_lop.csv"   # 例如 Qwen3 embedding 相似度

        # 1. 读取两个矩阵
        df1 = load_similarity_matrix(csv_file_1)
        df2 = load_similarity_matrix(csv_file_2)
        if 'answer' in csv_file_1:
            print("矩阵 1 来自答案分子相似度矩阵")
        else:
        # 2. （可选）检查行列标签是否一致
            if not df1.index.equals(df2.index) or not df1.columns.equals(df2.columns):
                print("⚠️ 警告：两个矩阵的行列标签不一致！")
                print("Matrix 1 labels:", df1.index.tolist())
                print("Matrix 2 labels:", df2.index.tolist())
                # 如果顺序不同但内容相同，可以按共同顺序对齐
                common_labels = df1.index.intersection(df2.index)
                if len(common_labels) < len(df1):
                    raise ValueError("行列标签不匹配，无法安全对齐。")
                df1 = df1.loc[common_labels, common_labels]
                df2 = df2.loc[common_labels, common_labels]

        # 3. 转为 NumPy 数组
        mat1 = df1.values.astype(float)
        mat2 = df2.values.astype(float)

        # 4. 计算 Spearman 相关
        rho, pval, n_pairs = spearman_upper_triangle(mat1, mat2)

        # 5. 输出结果
        print(f"✅ 成功比较 {n_pairs} 对分子相似度值")
        print(f"Spearman 相关系数 (ρ): {rho:.4f}")
        print(f"p 值: {pval:.2e}")
        
        if pval < 0.05:
            print("🟢 结果在 α=0.05 水平下显著")
        else:
            print("🔴 结果不显著")