import json

# 读取原始 JSON 文件
with open("/BioLatent/Bio-LatentCOT/refine_corrected/stage3_inference.proc0.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 过滤 test_results 中 task 不是 "qed" 的数据
filtered_results = [item for item in data.get("test_results", []) if item.get("task") == "solubility"]

# 生成新的 JSON 数据
new_data = {
    "timestamp": data.get("timestamp"),
    "test_data_path": data.get("test_data_path"),
    "model_info": data.get("model_info"),
    "generation_config": data.get("generation_config"),
    "num_samples": len(filtered_results),
    "test_results": filtered_results
}

# 保存到新文件
with open("filtered_non_qed.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)

print(f"已保存 {len(filtered_results)} 条 task 不是 'qed' 的样本到 filtered_non_qed.json")
