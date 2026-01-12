# ClinTox: CT_TOX
# HIV: HIV_active
# Tox21: just union them all

import pandas as pd
import numpy as np
import argparse
import json
import os

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

def preprocess_single_subtask(csv_path, output_path, task_name):
    df = pd.read_csv(csv_path)
    
    preprocessed = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        smiles = row[mol_col[task_name]]
        gt = False
        for label in gt_cols[task_name]:
            if (float)(row[label]) == 1:
                gt = True
                break
        
        prompt = prompt_templates[task_name].replace("[SMILES]", smiles)
        
        preprocessed.append({
            "query": prompt,
            "task": "property_prediction",
            "subtask": task_name,
            "gt": "Yes" if gt else "No",
            "meta": {
                "molecule": smiles
            }
        })
        
    with open(output_path, "w") as f:
        json.dump(preprocessed, f, indent=4)
        
def preprocess_property_prediction(csv_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for task in task_names:
        preprocess_single_subtask(os.path.join(csv_dir, f"{task}_test.csv"), os.path.join(output_dir, f"{task}.json"), task)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    preprocess_property_prediction(args.csv_dir, args.output_dir)