import pandas as pd
import numpy as np
import argparse
import json
import os

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import DataStructs
from utils import mol_to_canonical

prompt_template = '''You are a chemical assistant, your task is to predict the product SMILES. Given the reactant molecular SMILES, predict the product molecular SMILES based on your knowledge of chemical reaction mechanisms and organic synthesis.\n\nInput: Reactant SMILES string.\nOutput: Product SMILES string(s).\n\nYour final answer must be formatted as:\n<answer> SMILES </answer>\n\nInput reactant SMILES: [SMILES]'''

def preprocess_reaction_prediction(csv_path, output_root):
    df = pd.read_csv(csv_path)
    
    results = []
    
    for i in range(len(df)):
        # 提取整行
        row = df.iloc[i]
        reactant = row['reactant']
        product = row['product']
        
        prompt = prompt_template.replace('[SMILES]', reactant)
        
        results.append({
            "query": prompt,
            "task": "reaction_prediction",
            "subtask": "reaction_prediction",
            "gt": "",
            "meta":{
                "reactant": reactant,
                "reference": product
            }
        })
    
    os.makedirs(output_root, exist_ok=True)
    
    with open(f'{output_root}/reaction_prediction.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    preprocess_reaction_prediction(args.csv_path, args.output_dir)