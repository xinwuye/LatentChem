import os
import json
from collections import defaultdict
from typing import Any, Dict, List
import logging
import glob

logger = logging.getLogger(__name__)

def build_grouped_save_data(
    raw_results_path: str,
    save_results_dir: str,
    log_name: str
) -> Dict[str, Any]:

    base, ext = os.path.splitext(raw_results_path)
    ext = ext or ".json"
    dir_path = os.path.dirname(raw_results_path) or "."
    base_name = os.path.basename(base)

    # shard pattern: xxx.proc*.json
    shard_pattern = os.path.join(
        dir_path,
        f"{base_name}.proc*{ext}"
    )

    shard_files = sorted(glob.glob(shard_pattern))

    if shard_files:
        files_to_read = shard_files
    else:
        files_to_read = [raw_results_path]

    merged_results = []

    for path in files_to_read:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "test_results" not in data:
            raise KeyError(f"'test_results' not found in {path}")

        if not isinstance(data["test_results"], list):
            raise TypeError(f"'test_results' is not a list in {path}")

        merged_results.extend(data["test_results"])

    results = merged_results
    
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        task_key = r.get("task", "__unknown__")
        grouped[task_key].append(r)
        
    for task_key, items in grouped.items():
        items.sort(key=lambda x: x.get("sample_id", 0))

    for task, group in grouped.items():
        task_dir = os.path.join(save_results_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        task_path = os.path.join(task_dir, f'{log_name}.json')
        with open(task_path, 'w') as f:
            json.dump(group, f, ensure_ascii=False, indent=4)
            
build_grouped_save_data("/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval/Mol-Instructions/stage2_with_cot/mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage2_with_cot/inference/results_20260110-050429_p0.json","/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/data/ChemLLMBench_result","mytest")