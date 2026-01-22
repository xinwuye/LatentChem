import os
import json
from collections import defaultdict
from typing import Any, Dict, List
import logging
import glob
import re

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
            content = f.read().strip()
            # print(content)

            # Check if the file contains multiple JSON objects concatenated together
            # This is common when JSONL (JSON Lines) format is saved incorrectly
            try:
                # First, try to parse as a single JSON object
                data = json.loads(content)
                # print(data)
                # If successful, check if it has the expected structure
                if "test_results" in data:
                    merged_results.extend(data["test_results"])
                else:
                    raise KeyError("'test_results' not found in single JSON object")
            except json.JSONDecodeError:
                # If single JSON parsing fails, try to parse as multiple JSON objects
                logger.warning(f"Attempting to parse {path} as multiple JSON objects...")

                # Reset file pointer and read line by line to handle concatenated JSON
                f.seek(0)
                content = f.read()

                # Method 1: Try to split on "}\n{" which commonly separates JSON objects
                json_parts = re.split(r'\}\s*\n?\s*\{', content)

                # Process each potential JSON part
                for i, part in enumerate(json_parts):
                    # Clean up the part
                    part = part.strip()

                    # Add back braces if they're missing
                    if not part.startswith('{'):
                        part = '{' + part
                    if not part.endswith('}'):
                        part = part + '}'

                    try:
                        obj = json.loads(part)
                        # If this is the main object with test_results, use it
                        if "test_results" in obj:
                            merged_results.extend(obj["test_results"])
                            logger.info(f"Found and processed main results object in part {i}")
                        # If it's a different object, we might need to handle it differently
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse part {i} as JSON: {part[:100]}...")

                        # Alternative approach: Try to find JSON objects using character-by-character parsing
                        try:
                            # Look for complete JSON objects in the remaining content
                            start_pos = 0
                            while start_pos < len(part):
                                # Find the next opening brace
                                start_brace = part.find('{', start_pos)
                                if start_brace == -1:
                                    break

                                # Find the corresponding closing brace
                                brace_count = 0
                                pos = start_brace
                                while pos < len(part):
                                    if part[pos] == '{':
                                        brace_count += 1
                                    elif part[pos] == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            # Found a complete JSON object
                                            json_str = part[start_brace:pos+1]
                                            try:
                                                obj = json.loads(json_str)
                                                if "test_results" in obj:
                                                    merged_results.extend(obj["test_results"])
                                                    logger.info(f"Found and processed results from extracted JSON object")
                                                break
                                            except json.JSONDecodeError:
                                                pass
                                            break
                                    pos += 1

                                start_pos = pos + 1

                        except Exception as e:
                            logger.error(f"Error processing part {i}: {e}")

                # If we still haven't found any results, try another approach
                if not merged_results:
                    # Method 2: Try to parse the whole content as a stream of JSON objects
                    logger.info("Trying alternative parsing method for concatenated JSON...")
                    try:
                        # Attempt to parse as JSONL (JSON Lines) format
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines):
                            line = line.strip()
                            if line:
                                try:
                                    obj = json.loads(line)
                                    if "test_results" in obj:
                                        merged_results.extend(obj["test_results"])
                                        logger.info(f"Found results in line {line_num}")
                                        break  # Assuming the main object is in one line
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        logger.error(f"Error in alternative parsing: {e}")

                # If still no results, raise an error
                if not merged_results:
                    raise ValueError(f"Could not extract any valid 'test_results' from {path}")

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
build_grouped_save_data("/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/outputs/exp_chemllmbench_stage3_0_7_no/results/merge.json","/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/data/chemllmbench/", "inference_result_0_7_no")