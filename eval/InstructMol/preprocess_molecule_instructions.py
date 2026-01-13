"""
Preprocessing script for Molecule-Oriented Instructions dataset.

This script filters and converts the Molecule-Oriented Instructions dataset
to Bio-LatentCOT evaluation format.

Bio-LatentCOT supports:
- Molecule Description Generation (Mol2Cap)
- Chemical Reaction Analysis (forward_reaction_prediction, retrosynthesis, reagent_prediction)

Bio-LatentCOT does NOT support:
- description_guided_molecule_design (Cap2Mol) - requires generating molecules, not text
"""

import json
import os
import argparse
import selfies as sf
from pathlib import Path
from tqdm import tqdm


# Tasks supported by Bio-LatentCOT
SUPPORTED_TASKS = {
    'molecular_description_generation': {
        'eval_name': 'molecular_description_generation',
        'subtask': 'molecule_description_generation'
    },
    'forward_reaction_prediction': {
        'eval_name': 'forward_reaction_prediction',
        'subtask': 'forward_reaction_prediction'
    },
    'retrosynthesis': {
        'eval_name': 'retrosynthesis',
        'subtask': 'retrosynthesis'
    },
    'reagent_prediction': {
        'eval_name': 'reagent_prediction',
        'subtask': 'reagent_prediction'
    }
}


def selfies_to_smiles(selfies_str):
    """Convert SELFIES to SMILES. Returns None if conversion fails."""
    try:
        # Handle multiple molecules separated by '.'
        parts = selfies_str.split('.')
        smiles_parts = []
        for part in parts:
            if part.strip():
                smiles = sf.decoder(part.strip())
                if smiles:
                    smiles_parts.append(smiles)
        return '.'.join(smiles_parts) if smiles_parts else None
    except Exception as e:
        return None


def preprocess_molecular_description_generation(input_file, output_dir, max_samples=None):
    """
    Preprocess molecular description generation task.
    Input: SELFIES → Output: Text description
    """
    print(f"\nProcessing molecular_description_generation...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    output_data = []
    failed_conversions = 0

    for idx, item in enumerate(tqdm(data, desc="Converting SELFIES to SMILES")):
        instruction = item.get('instruction', '')
        input_selfies = item.get('input', '')
        output_text = item.get('output', '')
        metadata = item.get('metadata', {})

        # Convert SELFIES to SMILES
        smiles = selfies_to_smiles(input_selfies)
        if not smiles:
            failed_conversions += 1
            continue

        # Create the query (instruction)
        query = instruction.strip()

        # Create metadata
        meta = {
            "molecule": smiles,
            "gt": output_text
        }

        # Create a simple struct_cot
        struct_cot = {
            "output": output_text
        }

        # Format the entry
        entry = {
            "id": f"mol_instr_moldesc_{idx}",
            "query": query,
            "meta": json.dumps(meta),
            "struct_cot": json.dumps(struct_cot),
            "subtask": "molecule_description_generation"
        }

        output_data.append(entry)

    print(f"  Total samples: {len(data)}")
    print(f"  Successful conversions: {len(output_data)}")
    print(f"  Failed conversions: {failed_conversions}")

    # Save to JSON file
    output_file = os.path.join(output_dir, "molecular_description_generation.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved to {output_file}")
    return output_file


def preprocess_reaction_task(input_file, output_dir, task_name, max_samples=None):
    """
    Preprocess reaction tasks (forward_reaction_prediction, retrosynthesis, reagent_prediction).
    Input: SELFIES (reactants/products) → Output: SELFIES (products/reactants/reagents)
    """
    print(f"\nProcessing {task_name}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    output_data = []
    failed_conversions = 0

    for idx, item in enumerate(tqdm(data, desc="Converting SELFIES to SMILES")):
        instruction = item.get('instruction', '')
        input_selfies = item.get('input', '')
        output_selfies = item.get('output', '')
        metadata = item.get('metadata', {})

        # Convert input SELFIES to SMILES
        input_smiles = selfies_to_smiles(input_selfies)
        if not input_smiles:
            failed_conversions += 1
            continue

        # Convert output SELFIES to SMILES
        output_smiles = selfies_to_smiles(output_selfies)
        if not output_smiles:
            failed_conversions += 1
            continue

        # Create the query (instruction)
        query = instruction.strip()

        # Parse reaction components based on task
        if task_name == 'forward_reaction_prediction':
            # Input: reactants + reagents → Output: products
            meta = {
                "reactants": input_smiles.split('.'),
                "products": output_smiles.split('.'),
                "gt": output_smiles
            }
        elif task_name == 'retrosynthesis':
            # Input: product → Output: reactants
            meta = {
                "products": [input_smiles],
                "reactants": output_smiles.split('.'),
                "gt": output_smiles
            }
        elif task_name == 'reagent_prediction':
            # Input: reactants + products → Output: reagents
            # Parse input to separate reactants and products (typically separated by '>>')
            meta = {
                "reactants": input_smiles.split('.'),
                "reagents": output_smiles.split('.'),
                "gt": output_smiles
            }
        else:
            meta = {
                "molecule": input_smiles,
                "gt": output_smiles
            }

        # Create a simple struct_cot
        struct_cot = {
            "output": output_smiles
        }

        # Format the entry
        entry = {
            "id": f"mol_instr_{task_name}_{idx}",
            "query": query,
            "meta": json.dumps(meta),
            "struct_cot": json.dumps(struct_cot),
            "subtask": SUPPORTED_TASKS[task_name]['subtask']
        }

        output_data.append(entry)

    print(f"  Total samples: {len(data)}")
    print(f"  Successful conversions: {len(output_data)}")
    print(f"  Failed conversions: {failed_conversions}")

    # Save to JSON file
    output_file = os.path.join(output_dir, f"{task_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"  Saved to {output_file}")
    return output_file


def preprocess_all_tasks(input_dir, output_dir, max_samples_per_task=None):
    """
    Preprocess all supported tasks from Molecule-Oriented Instructions.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("Preprocessing Molecule-Oriented Instructions Dataset")
    print("="*80)

    processed_files = []

    for task_name, task_config in SUPPORTED_TASKS.items():
        input_file = os.path.join(input_dir, f"{task_name}.json")

        if not os.path.exists(input_file):
            print(f"\nWarning: {input_file} not found, skipping...")
            continue

        if task_name == 'molecular_description_generation':
            output_file = preprocess_molecular_description_generation(
                input_file, output_dir, max_samples_per_task
            )
        else:
            output_file = preprocess_reaction_task(
                input_file, output_dir, task_name, max_samples_per_task
            )

        processed_files.append(output_file)

    print("\n" + "="*80)
    print(f"Preprocessing complete! Processed {len(processed_files)} tasks.")
    print("="*80)

    return processed_files


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Molecule-Oriented Instructions for Bio-LatentCOT evaluation"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing Molecule-Oriented Instructions JSON files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed JSON files"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per task (for testing, use None for all)"
    )

    args = parser.parse_args()

    preprocess_all_tasks(args.input_dir, args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
