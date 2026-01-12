import pandas as pd
import numpy as np
import argparse
import json
import os

prompt_template = '''You are an expert chemist. Given the molecular requirements description, your task is to design a new molecule using your experienced chemical Molecular Design knowledge. \n Input: Molecular requirements description string. Output: Molecule SMILES string.\nYour final answer must be formatted as <answer> SMILES </answer>\n\nInput molecular requirements description string: [DESC]. \n'''

def preprocess_molecule_design(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    preprocessed = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        desc = row['description']
        smiles = row['SMILES']
        
        prompt = prompt_template.replace("[DESC]", desc)
        
        preprocessed.append({
            "query": prompt,
            "task": "molecule_design",
            "subtask": "molecule_design",
            "gt": "",
            "meta": {
                "reference": smiles
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
    output_path = os.path.join(args.output_dir, "molecule_design.json")
    preprocess_molecule_design(args.csv_path, output_path)