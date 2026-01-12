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

prompt_iupac2smiles_template = '''You are a chemical assistent. Given the molecular IUPAC name, help me predict the molecular SMILES using your experienced chemical molecular IUPAC name and SMILES knowledge. Input: Molecule IUPAC string. Output: Molecule SMILES string.\nYour final answer must be formatted as <answer> SMILES </answer>\n\nInput IUPAC string: [IUPAC].'''

prompt_smiles2iupac_template = '''You are a chemical assistent. Given the molecular SMILES, help me predict the molecular IUPAC name using your experienced chemical molecular SMILES and IUPAC knowledge. Input: Molecule SMILES string. Output: Molecule IUPAC string.\nYour final answer must be formatted as <answer> IUPAC </answer>\n\nInput SMILES string: [SMILES].'''

def preprocess_name_prediction(csv_path, output_root):
    df = pd.read_csv(csv_path)
    # 保证 iupac 列非空（与原脚本一致）
    df = df[~df['iupac'].isna()].reset_index(drop=True)

    # 提取label为test的列
    test = df[df['label'] == 'test'].reset_index(drop=True)

    iupac2smiles = []
    smiles2iupac = []
    
    # 逐行处理
    for i in range(len(test)):
        # 提取整行
        row = test.iloc[i]
        smiles = row['smiles']
        iupac = row['iupac']
        
        canonicalized = mol_to_canonical(smiles)
        
        prompt_iupac2smiles = prompt_iupac2smiles_template.replace('[IUPAC]', iupac)
        prompt_smiles2iupac = prompt_smiles2iupac_template.replace('[SMILES]', canonicalized)
        
        iupac2smiles.append({
            "query": prompt_iupac2smiles,
            "task": "name_prediction",
            "subtask": "iupac2smiles",
            "gt": "",
            "meta": {
                "reference": canonicalized,
            }
        })
        
        smiles2iupac.append({
            "query": prompt_smiles2iupac,
            "task": "name_prediction",
            "subtask": "smiles2iupac",
            "gt": "",
            "meta": {
                "reference": iupac,
                "molecule": canonicalized,
            }
        })
    
    os.makedirs(output_root, exist_ok=True)
    
    with open(f'{output_root}/iupac2smiles.json', 'w') as f:
        json.dump(iupac2smiles, f, indent=4)
    
    with open(f'{output_root}/smiles2iupac.json', 'w') as f:
        json.dump(smiles2iupac, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    preprocess_name_prediction(args.csv_path, args.output_dir)