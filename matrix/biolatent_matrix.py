#计算

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
#测试第几步
num=9
# 读取 JSONL 文件（每行一个 JSON 对象）
data = []
with open('/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval/silmilarity_matrix/sorted_data_logp.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:  # 跳过空行
            data.append(json.loads(line))

# 提取 smiles 和 latent_block
smiles_list = [item['smiles'][0] for item in data]
latent_blocks = [item['latent_block'] for item in data]#[num]

# 计算每个 latent_block 的平均值向量
def calculate_mean_vector(latent_block):
    """
    将 latent_block 中的多个子列表用其平均值表示为固定长度的向量。
    """
    vectors = [np.array(sublist) for sublist in latent_block]
    mean_vector = np.mean(np.vstack(vectors), axis=0)
    return mean_vector



# 为所有 latent_block 计算平均向量
mean_vectors = []
for block in latent_blocks:
    try:
        mean_vectors.append(calculate_mean_vector(block))
    except Exception as e:
        print(f"Error processing latent_block: {block}. Error: {e}")
        mean_vectors.append(np.zeros(len(mean_vectors[0]) if mean_vectors else 1))  # 填充零向量作为默认值

# 确保所有向量长度一致（填充或截断）
# max_length = max(len(vec) for vec in mean_vectors)
# padded_vectors = [np.pad(vec, (0, max_length - len(vec)), mode='constant') for vec in mean_vectors]

# 转换为 NumPy 矩阵
matrix = np.vstack(mean_vectors)
# 计算余弦相似度矩阵
similarity_matrix = cosine_similarity(matrix)

# 打印相似度矩阵
print("相似度矩阵：")
print(similarity_matrix)

# 保存到文件（CSV 格式）
df = pd.DataFrame(similarity_matrix, index=smiles_list, columns=smiles_list)
df.to_csv('/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval/silmilarity_matrix/bio_latent_similarity_matrix_logp'+'mean'+'.csv')

print("✅ 相似度矩阵已保存到 bio_latent_similarity_matrix_logp_mean.csv")