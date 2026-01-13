"""
Preprocessing script for ChEBI-20 dataset.

This script converts the ChEBI-20-MM CSV format into a format compatible
with Bio-LatentCOT evaluation pipeline (similar to ChemCotBench format).

The task is: Molecule Description Generation (Mol2Cap)
Input: SMILES string
Output: Natural language description
"""

import json
import csv
import os
import argparse
from pathlib import Path


def preprocess_chebi20(csv_path, output_dir, split='test'):
    """
    Convert ChEBI-20-MM CSV to Bio-LatentCOT eval format.

    Args:
        csv_path: Path to ChEBI-20-MM CSV file (test.csv, train.csv, or validation.csv)
        output_dir: Output directory for processed JSON files
        split: Dataset split name ('test', 'train', or 'validation')
    """
    os.makedirs(output_dir, exist_ok=True)

    output_data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader):
            cid = row.get('CID', str(idx))
            smiles = row.get('SMILES', '')
            description = row.get('description', '')

            if not smiles or not description:
                print(f"Warning: Skipping row {idx} due to missing SMILES or description")
                continue

            # Create the prompt for molecule description generation
            query = "Please provide a description of this molecule."

            # Create metadata
            meta = {
                "molecule": smiles,
                "gt": description,
                "cid": cid
            }

            # Create a simple struct_cot (no actual CoT for this task)
            struct_cot = {
                "output": description
            }

            # Format the entry
            entry = {
                "id": f"chebi20_{split}_{cid}",
                "query": query,
                "meta": json.dumps(meta),
                "struct_cot": json.dumps(struct_cot),
                "subtask": "molecule_description_generation"
            }

            output_data.append(entry)

    # Save to JSON file
    output_file = os.path.join(output_dir, f"chebi20_{split}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(output_data)} samples from {csv_path}")
    print(f"Saved to {output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Preprocess ChEBI-20 dataset for Bio-LatentCOT evaluation")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to ChEBI-20-MM CSV file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed JSON files")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "train", "validation"],
                        help="Dataset split name")

    args = parser.parse_args()

    preprocess_chebi20(args.csv_path, args.output_dir, args.split)


if __name__ == "__main__":
    main()
