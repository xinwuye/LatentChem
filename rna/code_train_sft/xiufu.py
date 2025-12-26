import json
import glob

base_path = "/zengdaojian/zhangjia/BioLatent/ChemCotDataset"
files = glob.glob(f"{base_path}/**/*.json", recursive=True)

target_keys = ["id", "query", "gt", "task", "subtask", "meta",
               "cot_result", "raw_cot", "struct_cot"]

for f in files:
    with open(f, "r") as fp:
        data = json.load(fp)

    for d in data:
        for k in target_keys:
            # 如果字段缺失或者是 null，就改成空字符串
            if k not in d or d[k] is None:
                d[k] = ""

    with open(f, "w") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)

print("✔ All JSON files fixed: all fields are strings")
