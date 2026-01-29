import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from extract_molecules import extract_source_molecules
import json
# 1. 加载 tokenizer 和 base 模型（注意：必须是 base，不是 chat）
model_name = "/BioLatent/Bio-LatentCOT/models/Qwen3-8B-Base"  # 确保你有权访问此模型

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",          # 自动分配 GPU
    torch_dtype=torch.bfloat16  # 节省显存（也可用 float16）
)

model.eval()  # 设置为评估模式

# 2. 分子 SMILES 列表
json_file_path = "/BioLatent/Bio-LatentCOT/data/ChemCoTBench/chemcotbench/mol_opt/logp.json"

# Extract all source molecules
smiles_list = extract_source_molecules(json_file_path)

# 3. 提取每个 SMILES 的 embedding（使用 [CLS] 或平均池化）
def get_smiles_embedding(smiles: str, tokenizer, model, device="cuda"):
    # 添加特殊 token（Qwen 使用 </s> 作为 EOS）
    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        # 方法1: 使用最后一层所有 token 的平均（推荐用于短文本）
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        # 对有效 token 求平均
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1, keepdim=True)
        mean_embedding = sum_embeddings / sum_mask.clamp(min=1e-9)
        return mean_embedding.squeeze().cpu().numpy()

# 4. 为所有 SMILES 提取 embedding
embeddings = []
# for smi in smiles_list:
#     emb = get_smiles_embedding(smi, tokenizer, model)
#     embeddings.append(emb)
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
for item in data:
    smi = item['query']
    emb = get_smiles_embedding(smi, tokenizer, model)
    embeddings.append(emb)
embeddings = np.array(embeddings)  # shape: (n_mols, hidden_size)
print("Embedding shape:", embeddings.shape)

# 计算余弦相似度矩阵
sim_matrix = cosine_similarity(embeddings)

# 转为 DataFrame 并保存
df_sim = pd.DataFrame(
    sim_matrix,
    index=smiles_list,
    columns=smiles_list
)

# 保存到 CSV
df_sim.to_csv("/BioLatent/Bio-LatentCOT/eval/silmilarity_matrix/qwen3_smiles_similarity_string_lop.csv", float_format="%.4f")
print("相似度矩阵已保存到 qwen3_smiles_similarity_lop.csv")
print(df_sim)