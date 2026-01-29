import json
from extract_molecules import extract_source_molecules
# 给定的 SMILES 排序列表
# 2. 分子 SMILES 列表
json_file_path = "/BioLatent/Bio-LatentCOT/data/ChemCoTBench/chemcotbench/mol_opt/logp.json"

# Extract all source molecules
smiles_list = extract_source_molecules(json_file_path)

# 读取原始 JSONL 文件（每行一个 JSON 对象）
json_objects = []
with open('/BioLatent/Bio-LatentCOT/new_latent/stage3_latent_logp.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)
            json_objects.append(obj)

# 创建一个映射：smiles -> 原始对象 + 索引（用于排序）
smiles_to_obj = {obj['smiles'][0]: obj for obj in json_objects}

# 按 smiles_order 排序
sorted_objects = [smiles_to_obj[smi] for smi in smiles_list if smi in smiles_to_obj]

# 输出到新文件（可选：保留为 JSONL 格式）
with open('/BioLatent/Bio-LatentCOT/eval/silmilarity_matrix/sorted_data_logp.jsonl', 'w', encoding='utf-8') as f:
    for obj in sorted_objects:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

print("✅ 排序完成！结果已保存到 sorted_data_qed.jsonl")