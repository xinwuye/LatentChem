import json
import os
from pathlib import Path

def merge_json_files(results_dir, output_file):
    """
    Merges all JSON files in the results directory into a single file.
    
    Args:
        results_dir (str): Path to the directory containing JSON files to merge
        output_file (str): Path to the output file where merged results will be saved
    """
    results_path = Path(results_dir)
    
    # Find all JSON files in the results directory, excluding the output file
    all_json_files = list(results_path.glob("*.json"))
    json_files = [f for f in all_json_files if f.name != os.path.basename(output_file)]
    
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to merge:")
    for file in json_files:
        print(f"  - {file.name}")
    
    # Sort files to ensure consistent ordering
    json_files.sort()
    
    merged_data = []
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Handle different data formats
                if isinstance(data, list):
                    # If the file contains a list, extend the merged_data
                    merged_data.extend(data)
                elif isinstance(data, dict):
                    # If the file contains a single object, append it
                    merged_data.append(data)
                else:
                    print(f"Warning: Unexpected data type in {json_file.name}: {type(data)}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {json_file.name}: {e}")
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    # Write merged data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"Merged {len(json_files)} files into {output_file}")
    print(f"Total entries in merged file: {len(merged_data)}")


def main():
    import sys
    # Allow specifying directory as command line argument, otherwise use default
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "/BioLatent/Bio-LatentCOT/outputs/exp_chemllmbench_stage3_1_5_no/results"

    # Generate output filename based on input directory
    parent_dir = os.path.dirname(results_dir.rstrip('/'))
    output_file = os.path.join(parent_dir, 'merged_results.json')

    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        return

    merge_json_files(results_dir, output_file)


if __name__ == "__main__":
    main()