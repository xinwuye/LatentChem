import os
import json
from glob import glob

def merge_json_folder(input_folder: str, output_file: str):
    """
    合并一个文件夹下的所有 JSON 文件
    - test_results 列表合并
    - 其他字段取第一个文件的
    """
    input_folder = os.path.abspath(input_folder)
    if not os.path.isdir(input_folder):
        raise ValueError(f"Folder not found: {input_folder}")

    json_files = sorted(glob(os.path.join(input_folder, "*.json")))
    if not json_files:
        raise ValueError(f"No JSON files found in folder: {input_folder}")

    merged_data = {}
    merged_test_results = []

    for idx, fpath in enumerate(json_files):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "test_results" not in data or not isinstance(data["test_results"], list):
            raise KeyError(f"'test_results' missing or not a list in {fpath}")

        merged_test_results.extend(data["test_results"])

        if idx == 0:
            # 其他字段只取第一个文件的
            for k, v in data.items():
                if k != "test_results":
                    merged_data[k] = v

    merged_data["test_results"] = merged_test_results

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"Merged {len(json_files)} files from {input_folder} into {output_file}")
    print(f"Total test_results: {len(merged_test_results)}")


# ==========================
# 示例用法
# ==========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge JSON files in a folder")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing JSON files")
    parser.add_argument("--output_file", type=str, required=True, help="Output merged JSON file path")
    args = parser.parse_args()

    merge_json_folder(args.input_folder, args.output_file)
