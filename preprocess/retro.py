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

prompt_template = '''You are a chemical assistant, your task is to predict the reactants SMILES. Given the product SMILES, predict the reactant molecular SMILES based on your experienced chemical Retrosynthesis knowledge.\n\nInput: Reactant SMILES string.\nOutput: Product SMILES string(s).\n\nYour final answer must be formatted as:\n<answer> SMILES </answer>\n\nInput product SMILES: [SMILES]'''

def preprocess_retro(csv_path, output_root):
    df = pd.read_csv(csv_path)
    
    results = []
    
    for i in range(len(df)):
        # 提取整行
        row = df.iloc[i]
        reactant = row['reactants_smiles']
        product = row['products_smiles']
        
        prompt = prompt_template.replace('[SMILES]', product)
        
        results.append({
            "query": prompt,
            "task": "retro",
            "subtask": "retro",
            "gt": "",
            "meta":{
                "reference": reactant,
                "product": product
            }
        })
    
    os.makedirs(output_root, exist_ok=True)
    
    with open(f'{output_root}/retro.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    preprocess_retro(args.csv_path, args.output_dir)