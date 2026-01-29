#!/usr/bin/env python
"""
Preprocessing script for ChemLLMBench dataset
Combines all data files from different tasks into a single directory with standardized format
"""

import pandas as pd
import numpy as np
import argparse
import json
import os
import glob
from pathlib import Path


def preprocess_molecule_captioning(csv_path, output_root):
    """Preprocess molecule captioning data (SMILES -> description)"""
    df = pd.read_csv(csv_path)

    results = []
    prompt_template = '''You are an expert chemist. Given the molecular SMILES, your task is to provide the detailed description of the molecule using your experienced chemical Molecular knowledge. \n Input: Molecule SMILES string. Output: Molecular detailed description string.\nYour final answer must be formatted as <answer> Description </answer>\n\nInput molecular SMILES string: [SMILES]. \n'''

    for i in range(len(df)):
        row = df.iloc[i]
        desc = row['description']
        smiles = row['SMILES']

        prompt = prompt_template.replace("[SMILES]", smiles)

        results.append({
            "id": f"molecule_captioning_{i}",
            "query": prompt,
            "task": "molecule_captioning",
            "subtask": "molecule_captioning",
            "gt": desc,
            "reference": desc,
            "meta": {
                "molecule": smiles
            }
        })

    # Create subdirectory for molecule_captioning
    task_output_dir = os.path.join(output_root, "molecule_captioning")
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, "molecule_captioning.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(results)} molecule captioning samples to {output_path}")


def preprocess_molecule_design(csv_path, output_root):
    """Preprocess molecule design data (description -> SMILES)"""
    df = pd.read_csv(csv_path)

    results = []
    prompt_template = '''You are an expert chemist. Given the molecular description, your task is to design the corresponding molecular SMILES string using your experienced chemical knowledge. \n Input: Molecular description string. Output: SMILES string.\nYour final answer must be formatted as <answer> SMILES </answer>\n\nInput molecular description: [DESCRIPTION]. \n'''

    for i in range(len(df)):
        row = df.iloc[i]
        desc = row['description']
        smiles = row['SMILES']

        prompt = prompt_template.replace("[DESCRIPTION]", desc)

        results.append({
            "id": f"molecule_design_{i}",
            "query": prompt,
            "task": "molecule_design",
            "subtask": "molecule_design",
            "gt": smiles,
            "reference": smiles,
            "meta": {
                "molecule": desc
            }
        })

    # Create subdirectory for molecule_design
    task_output_dir = os.path.join(output_root, "molecule_design")
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, "molecule_design.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(results)} molecule design samples to {output_path}")


def preprocess_reaction_prediction(csv_path, output_root):
    """Preprocess reaction prediction data (reactant -> product)"""
    df = pd.read_csv(csv_path)

    results = []
    prompt_template = '''You are a chemical assistant, your task is to predict the product SMILES. Given the reactant molecular SMILES, predict the product molecular SMILES based on your knowledge of chemical reaction mechanisms and organic synthesis.\n\nInput: Reactant SMILES string.\nOutput: Product SMILES string(s).\n\nYour final answer must be formatted as:\n<answer> SMILES </answer>\n\nInput reactant SMILES: [SMILES]'''

    for i in range(len(df)):
        row = df.iloc[i]
        reactant = row['reactant']
        product = row['product']

        prompt = prompt_template.replace('[SMILES]', reactant)

        results.append({
            "id": f"reaction_prediction_{i}",
            "query": prompt,
            "task": "reaction_prediction",
            "subtask": "reaction_prediction",
            "gt": product,
            "reference": product,
            "meta": {
                "molecule": reactant
            }
        })

    # Create subdirectory for reaction_prediction
    task_output_dir = os.path.join(output_root, "reaction_prediction")
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, "reaction_prediction.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(results)} reaction prediction samples to {output_path}")


def preprocess_reagent_selection(csv_path, output_dir):
    """Preprocess reagent selection data (reaction -> reagent)"""
    df = pd.read_csv(csv_path)

    results = []
    prompt_template = '''You are an expert chemist. Given the chemical reaction, your task is to select the appropriate reagent from the options provided. \n Input: Reaction SMILES string. Output: Reagent SMILES string.\nYour final answer must be formatted as <answer> Reagent SMILES </answer>\n\nInput reaction SMILES: [REACTION]. \n'''

    for i in range(len(df)):
        row = df.iloc[i]
        reaction = row['reaction'] if 'reaction' in row else row['reactant'] if 'reactant' in row else row.iloc[0]  # fallback to first column
        reagent = row['reagent'] if 'reagent' in row else row['product'] if 'product' in row else row.iloc[1]  # fallback to second column

        prompt = prompt_template.replace("[REACTION]", str(reaction))

        results.append({
            "id": f"reagent_selection_{i}",
            "query": prompt,
            "task": "reagent_selection",
            "subtask": "reagent_selection",
            "gt": str(reagent),
            "reference": str(reagent),
            "meta": {
                "molecule": str(reaction)
            }
        })

    # Create subdirectory for reagent_selection
    task_output_dir = os.path.join(output_dir, "reagent_selection")
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, "reagent_selection.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(results)} reagent selection samples to {output_path}")


def preprocess_retro(csv_path, output_dir):
    """Preprocess retrosynthesis data (product -> reactant)"""
    df = pd.read_csv(csv_path)

    results = []
    prompt_template = '''You are a chemical assistant, your task is to predict the reactants SMILES. Given the product SMILES, predict the reactant molecular SMILES based on your experienced chemical Retrosynthesis knowledge.\n\nInput: Product SMILES string.\nOutput: Reactant SMILES string(s).\n\nYour final answer must be formatted as:\n<answer> SMILES </answer>\n\nInput product SMILES: [SMILES]'''

    for i in range(len(df)):
        row = df.iloc[i]
        reactant = row['reactant'] if 'reactant' in row else row['reactants_smiles']
        product = row['product'] if 'product' in row else row['products_smiles']

        prompt = prompt_template.replace('[SMILES]', product)

        results.append({
            "id": f"retro_{i}",
            "query": prompt,
            "task": "retro",
            "subtask": "retro",
            "gt": reactant,
            "reference": reactant,
            "meta": {
                "molecule": product
            }
        })

    # Create subdirectory for retro
    task_output_dir = os.path.join(output_dir, "retro")
    os.makedirs(task_output_dir, exist_ok=True)
    output_path = os.path.join(task_output_dir, "retro.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(results)} retrosynthesis samples to {output_path}")


def preprocess_property_prediction(csv_path, output_dir):
    """Preprocess property prediction data (SMILES -> property)"""
    # Import required modules for property prediction
    import pandas as pd
    import os

    # Define constants similar to the property_prediction.py file
    task_names = ["BACE", "BBBP", "ClinTox", "HIV", "Tox"]

    mol_col = {
        "BACE": "mol",
        "BBBP": "smiles",
        "ClinTox": "smiles",
        "HIV": "smiles",
        "Tox": "smiles",
    }

    gt_cols = {
        "BACE": ["Class"],
        "BBBP": ["p_np"],
        "ClinTox": ["CT_TOX"],
        "HIV": ["HIV_active"],
        "Tox": ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
    }

    prompt_templates = {
        "BBBP": "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically penetration/non-penetration to the brain-blood barrier, based on the SMILES string representation of each molecule. You will be provided with several examples molecules, each accompanied by a binary label indicating whether it has penetrative property (Yes) or not (No). Input: Molecule SMILES string. Output: Yes / No.\nYour final answer must be formatted as <answer> Output: Yes / No </answer>\n\nInput SMILES string: [SMILES].\n",

        "BACE": "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. Given the SMILES string of a molecule, predict the molecular properties of a given chemical compound based on its structure, by analyzing wether it can inhibit(Yes) the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) or cannot inhibit(No) BACE1. Consider factors such as molecular weight, atom count, bond types, and functional groups in order to assess the compound's drug-likeness and its potential to serve as an effective therapeutic agent for Alzheimer's disease. Input: Molecule SMILES string. Output: Yes / No.\nYour final answer must be formatted as <answer> Output: Yes / No </answer>\n\nInput SMILES string: [SMILES]. \n",

        "Tox": "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is toxic(Yes) or Not toxic(No), based on the SMILES string representation of each molecule. A template will be provided. Input: Molecule SMILES string. Output: Yes / No.\nYour final answer must be formatted as <answer> Output: Yes / No </answer>\n\nInput SMILES string: [SMILES].\n",

        "HIV": "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge. Given the SELFIES string of a molecule, the task focuses on predicting molecular properties, specifically inhibit of HIV replication based on the SELFIES string representation of each molecule. You will be provided with several examples molecules, each accompanied by a binary label indicating whether a molecule can inhibit (Yes) or cannot inhibit (No) HIV replication. Additionally, the activity test results of the molecules are provided. There are three classes of the activity test: 1). CA: confirmed active, 2). CM: Confirmed moderately active 3.) CI: Confirmed inactive. The task is to precisely predict the binary label for a given molecule and its HIV activity test, considering its properties and its potential to impede HIV replication. Input: Molecule SMILES string. Output: Yes / No.\nYour final answer must be formatted as <answer> Output: Yes / No </answer>\n\nInput SMILES string: [SMILES].\n",

        "ClinTox": "You are an expert chemist, your task is to predict the property of molecule using your experienced chemical property prediction knowledge.\n Given the SMILES string of a molecule, the task focuses on predicting molecular properties, specifically wether a molecule is Clinically-trail-Toxic(Yes) or Not Clinically-trail-toxic (No) based on the SMILES string representation of each molecule.. The FDA-approved status will specify if the drug is approved by the FDA for clinical trials(Yes) or Not approved by the FDA for clinical trials(No). You will be provided with task template. Input: Molecule SMILES string. Output: Yes / No.\nYour final answer must be formatted as <answer> Output: Yes / No </answer>\n\nInput SMILES string: [SMILES].\n"
    }

    # Create subdirectory for property_prediction
    prop_output_dir = os.path.join(output_dir, "property_prediction")
    os.makedirs(prop_output_dir, exist_ok=True)

    # Process each task separately
    for task in task_names:
        task_csv_path = os.path.join(os.path.dirname(csv_path), f"{task}_test.csv")
        if os.path.exists(task_csv_path):
            df = pd.read_csv(task_csv_path)

            results = []

            for i in range(len(df)):
                row = df.iloc[i]
                smiles = row[mol_col[task]]
                gt = False
                for label in gt_cols[task]:
                    if label in row and float(row[label]) == 1:
                        gt = True
                        break

                prompt = prompt_templates[task].replace("[SMILES]", smiles)

                results.append({
                    "id": f"property_prediction_{task}_{i}",
                    "query": prompt,
                    "task": "property_prediction",
                    "subtask": task,
                    "gt": "Yes" if gt else "No",
                    "reference": "Yes" if gt else "No",
                    "meta": {
                        "molecule": smiles
                    }
                })

            output_path = os.path.join(prop_output_dir, f"{task}.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Processed {len(results)} {task} property prediction samples to {output_path}")


def preprocess_name_prediction(csv_path, output_dir):
    """Preprocess name prediction data (IUPAC <-> SMILES)"""
    import pandas as pd
    import os
    import json

    # Import RDKit and utility functions if available
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem import DataStructs

        def mol_to_canonical(smiles):
            """Simple canonicalization function if utils module is not available"""
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol)
            return smiles
    except ImportError:
        def mol_to_canonical(smiles):
            """Fallback canonicalization function"""
            return smiles

    prompt_iupac2smiles_template = '''You are a chemical assistent. Given the molecular IUPAC name, help me predict the molecular SMILES using your experienced chemical molecular IUPAC name and SMILES knowledge. Input: Molecule IUPAC string. Output: Molecule SMILES string.\nYour final answer must be formatted as <answer> SMILES </answer>\n\nInput IUPAC string: [IUPAC].'''

    prompt_smiles2iupac_template = '''You are a chemical assistent. Given the molecular SMILES, help me predict the molecular IUPAC name using your experienced chemical molecular SMILES and IUPAC knowledge. Input: Molecule SMILES string. Output: Molecule IUPAC string.\nYour final answer must be formatted as <answer> IUPAC </answer>\n\nInput SMILES string: [SMILES].'''

    df = pd.read_csv(csv_path)
    # Filter out rows where iupac is NaN
    df = df[~df['iupac'].isna()].reset_index(drop=True)

    # Extract rows labeled as 'test'
    test = df[df['label'] == 'test'].reset_index(drop=True)

    iupac2smiles = []
    smiles2iupac = []

    # Process each test row
    for i in range(len(test)):
        row = test.iloc[i]
        smiles = row['smiles']
        iupac = row['iupac']

        canonicalized = mol_to_canonical(smiles)

        prompt_iupac2smiles = prompt_iupac2smiles_template.replace('[IUPAC]', iupac)
        prompt_smiles2iupac = prompt_smiles2iupac_template.replace('[SMILES]', canonicalized)

        iupac2smiles.append({
            "id": f"name_prediction_iupac2smiles_{i}",
            "query": prompt_iupac2smiles,
            "task": "name_prediction",
            "subtask": "iupac2smiles",
            "gt": canonicalized,
            "reference": canonicalized,
            "meta": {
                "molecule": iupac
            }
        })

        smiles2iupac.append({
            "id": f"name_prediction_smiles2iupac_{i}",
            "query": prompt_smiles2iupac,
            "task": "name_prediction",
            "subtask": "smiles2iupac",
            "gt": iupac,
            "reference": iupac,
            "meta": {
                "molecule": canonicalized,
            }
        })

    # Create subdirectory for name_prediction
    name_pred_output_dir = os.path.join(output_dir, "name_prediction")
    os.makedirs(name_pred_output_dir, exist_ok=True)

    with open(os.path.join(name_pred_output_dir, 'iupac2smiles.json'), 'w') as f:
        json.dump(iupac2smiles, f, indent=4)
    print(f"Processed {len(iupac2smiles)} IUPAC to SMILES samples")

    with open(os.path.join(name_pred_output_dir, 'smiles2iupac.json'), 'w') as f:
        json.dump(smiles2iupac, f, indent=4)
    print(f"Processed {len(smiles2iupac)} SMILES to IUPAC samples")


def combine_all_tasks(input_dir, output_dir):
    """Process all task-specific data files and put them in subdirectories"""
    print(f"Processing ChemLLMBench data from {input_dir} to {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define task processors with correct names
    task_processors = {
        'molecule_captioning': preprocess_molecule_captioning,
        'molecule_design': preprocess_molecule_design,
        'reaction_prediction': preprocess_reaction_prediction,
        'reagent_selection': preprocess_reagent_selection,
        'retro': preprocess_retro,
        'property_prediction': preprocess_property_prediction,
        'name_prediction': preprocess_name_prediction
    }

    # Process each task directory
    for task_name, processor_func in task_processors.items():
        task_dir = os.path.join(input_dir, task_name)
        if os.path.exists(task_dir):
            print(f"Processing task: {task_name}")

            # Find all CSV files in the task directory
            csv_files = glob.glob(os.path.join(task_dir, "*.csv"))

            if csv_files:
                # Use the first CSV file found (usually test file)
                csv_file = csv_files[0]
                print(f"  Found CSV file: {csv_file}")

                try:
                    # Pass the output_dir to the processor function, which will handle subdirectory creation
                    processor_func(csv_file, output_dir)
                except Exception as e:
                    print(f"  Error processing {task_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                print(f"  No CSV files found in {task_dir}")
        else:
            print(f"  Task directory does not exist: {task_dir}")


def load_inference_data(data_path, exclude_tasks=None):
    """
    Load data for inference from ChemLLMBench dataset
    Mimics the structure from dataloader.py for inference purposes
    """
    if exclude_tasks is None:
        exclude_tasks = ['molecule_caption', 'molecule_design']  # Common tasks to exclude for inference

    # Import necessary modules
    import json
    import os
    import glob
    from datasets import Dataset
    from datasets import concatenate_datasets

    print(f"Loading inference data from {data_path}")
    print(f"Excluding tasks: {exclude_tasks}")

    all_json_files = glob.glob(os.path.join(data_path, "**/*.json"), recursive=True)

    def filter_data(f):
        return all([not f.endswith(f"{task}.json") for task in exclude_tasks])

    data_files = [f for f in all_json_files if filter_data(f)]

    print(f"Found {len(data_files)} JSON files to process")

    if not data_files:
        print("No data files found matching the criteria")
        return None

    datasets_list = []

    for file_path in data_files:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Normalize the 'meta' field to ensure consistent schema across all files
        normalized_data = []
        for item in data:
            # Ensure all expected fields exist in meta to maintain consistent schema
            meta = item.get('meta', {})
            if not isinstance(meta, dict):
                meta = {'molecule': meta}  # If meta is not a dict, wrap it

            # Ensure consistent schema by defining ALL possible fields with consistent types
            normalized_meta = {
                'molecule': '',
                'reactants': [],
                'reagents': [],
                'products': [],
                'candidate_rank': '',
                'reference': ''
            }

            # Copy values from original meta, ensuring correct types
            for key, default_value in normalized_meta.items():
                if key in meta:
                    original_value = meta[key]
                    # Ensure type consistency
                    if isinstance(default_value, list):
                        # Convert to list if it's not already, ensuring all elements are strings
                        if isinstance(original_value, list):
                            # Filter out None values and convert all elements to strings
                            normalized_meta[key] = [str(item) for item in original_value if item is not None]
                        elif original_value is None:
                            normalized_meta[key] = []
                        else:
                            # If it's a string or other type, try to convert appropriately
                            normalized_meta[key] = [str(original_value)] if str(original_value) != '' else []
                    else:  # string type
                        if original_value is not None:
                            normalized_meta[key] = str(original_value)
                        else:
                            normalized_meta[key] = ''
                else:
                    # Use default value if key doesn't exist
                    normalized_meta[key] = default_value

            # Update the item with normalized meta
            item['meta'] = normalized_meta
            normalized_data.append(item)

        # Ensure all items have the same top-level schema before creating the dataset
        if normalized_data:
            # Define the expected top-level schema - based on the error, we need to include all possible fields
            expected_fields = {'id', 'query', 'struct_cot', 'meta', 'subtask', 'task', 'gt', 'raw_cot', 'cot_result'}

            # Add any missing fields with default values to all items
            for item in normalized_data:
                for field in expected_fields:
                    if field not in item:
                        if field == 'meta':
                            item[field] = {'molecule': '', 'reactants': [], 'reagents': [], 'products': [], 'candidate_rank': '', 'reference': ''}
                        elif field in ['id', 'query', 'struct_cot', 'subtask', 'task', 'gt', 'raw_cot', 'cot_result']:
                            item[field] = ''

        # Convert to HuggingFace Dataset
        single_ds = Dataset.from_list(normalized_data)
        datasets_list.append(single_ds)

    # Check if datasets_list is empty before concatenating
    if not datasets_list:
        print(f"Warning: No datasets found in {data_path} with the specified exclude_tasks {exclude_tasks}. Returning None.")
        return None
    else:
        # Ensure all datasets have the same schema before concatenating
        # Cast all datasets to have consistent schema
        from datasets import Dataset, Features, Value, Sequence
        # Define a consistent schema for all datasets
        consistent_schema = Features({
            'id': Value('string'),
            'query': Value('string'),
            'struct_cot': Value('string'),
            'raw_cot': Value('string'),
            'subtask': Value('string'),
            'task': Value('string'),  # Added based on error
            'gt': Value('string'),    # Added based on error
            'cot_result': Value('string'),
            'meta': {
                'candidate_rank': Value('string'),
                'molecule': Value('string'),
                'products': Sequence(Value('string')),
                'reactants': Sequence(Value('string')),
                'reagents': Sequence(Value('string')),
                'reference': Value('string')
            }
        })

        # Reformat each dataset to have the consistent schema
        reformatted_datasets = []
        for ds in datasets_list:
            # Convert dataset to pandas, then reconstruct with consistent schema
            try:
                df = ds.to_pandas()
                # Ensure all list columns have consistent types
                for idx, row in df.iterrows():
                    meta = row['meta']
                    # Ensure all list fields in meta are proper lists of strings
                    if not isinstance(meta.get('reactants', []), list):
                        df.at[idx, 'meta']['reactants'] = []
                    if not isinstance(meta.get('reagents', []), list):
                        df.at[idx, 'meta']['reagents'] = []
                    if not isinstance(meta.get('products', []), list):
                        df.at[idx, 'meta']['products'] = []

                    # Ensure all list elements are strings
                    df.at[idx, 'meta']['reactants'] = [str(r) for r in meta.get('reactants', []) if r is not None]
                    df.at[idx, 'meta']['reagents'] = [str(r) for r in meta.get('reagents', []) if r is not None]
                    df.at[idx, 'meta']['products'] = [str(p) for p in meta.get('products', []) if p is not None]

                # Create new dataset with consistent schema
                reformatted_ds = Dataset.from_pandas(df, features=consistent_schema)
                reformatted_datasets.append(reformatted_ds)
            except Exception as e:
                print(f"Error processing dataset: {e}")
                raise

        # Now concatenate the reformatted datasets
        ds = concatenate_datasets(reformatted_datasets)
        print(f"Successfully loaded and concatenated {len(datasets_list)} datasets with {len(ds)} total samples")
        return ds


def main():
    parser = argparse.ArgumentParser(description='Preprocess ChemLLMBench data')
    parser.add_argument('--input_dir', type=str, required=False, help='Input directory containing task subdirectories',default="/BioLatent/ChemLLMBench/data")
    parser.add_argument('--output_dir', type=str, required=False, help='Output directory to save processed data',default="/BioLatent/Bio-LatentCOT/data/ChemLLMBench")
    parser.add_argument('--load_inference', action='store_true', help='Load data for inference instead of preprocessing')

    args = parser.parse_args()

    if args.load_inference:
        # Load data for inference
        inference_data = load_inference_data(args.output_dir)
        if inference_data is not None:
            print(f"Inference data loaded successfully with {len(inference_data)} samples")
            # Print example of first sample if available
            if len(inference_data) > 0:
                print("Example of first inference sample:")
                print(inference_data[0])
        else:
            print("Failed to load inference data")
    else:
        # Perform preprocessing as usual
        combine_all_tasks(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()