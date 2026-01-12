import pandas as pd
import numpy as np
import argparse
import json
import os

prompt_template = '''You are an expert chemist. Given the molecular SMILES, your task is to provide the detailed description of the molecule using your experienced chemical Molecular knowledge. \n Input: Molecule SMILES string. Output: Molecular detailed description string.\nYour final answer must be formatted as <answer> Description </answer>\n\nInput molecular SMILES string: [SMILES]. \n'''

def preprocess_molecule_captioning(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    preprocessed = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        desc = row['description']
        smiles = row['SMILES']
        
        prompt = prompt_template.replace("[SMILES]", smiles)
        
        preprocessed.append({
            "query": prompt,
            "task": "molecule_captioning",
            "subtask": "molecule_captioning",
            "gt": desc,
            "meta": {
                "molecule": smiles
            }
        })
        
    with open(output_path, "w") as f:
        json.dump(preprocessed, f, indent=4)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "molecule_captioning.json")
    preprocess_molecule_captioning(args.csv_path, output_path)