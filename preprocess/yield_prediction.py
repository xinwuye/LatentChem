import pandas as pd
import numpy as np
import argparse
import json
import os

task_names = ["BH", "SU"]

def parse_reaction_smiles(reaction_smiles: str):
    if ">>" not in reaction_smiles:
        raise ValueError("not a reaction SMILES '>>'")

    left, right = reaction_smiles.split(">>", 1)

    reactants = [s.strip() for s in left.split(".") if s.strip()]
    products = [s.strip() for s in right.split(".") if s.strip()]

    return reactants, products

prompt_templates = {
    "BH": '''Given the reactants SMILES and catalyst/reagent SMILES, your task is to predict whether the following Buchwald–Hartwig coupling reaction is high-yielding based on your expert knowledge of palladium-catalyzed cross-coupling chemistry.\n\nA reaction is considered high-yielding if the isolated yield is greater than 70%.\n\nInput:\nReactants, catalysts and reagents SMILES and Products SMILES.\n\nOutput:\nYes if the reaction yield is >70%, \n No otherwise. Your final answer must be formatted as <answer> Yes / No </answer>. Input reactants, catalysts and reagents SMILES: [LEFT], products SMILES: [RIGHT]\n\n''',
    "SU": '''Given the reactants SMILES and catalyst/reagent SMILES, your task is to predict whether the following Suzuki–Miyaura coupling reaction is high-yielding based on your expert knowledge of palladium-catalyzed cross-coupling chemistry.\n\nA reaction is considered high-yielding if the isolated yield is greater than 70%.\n\nInput:\nReactants, catalysts and reagents SMILES and Products SMILES.\n\nOutput:\nYes if the reaction yield is >70%, \n No otherwise. Your final answer must be formatted as <answer> Yes / No </answer>. Input reactants, catalysts and reagents SMILES: [LEFT], products SMILES: [RIGHT]\n\n'''
}

def preprocess_single_subtask(npz_path, output_path, task_name):
    data = np.load(npz_path, allow_pickle=True)
    arr = data["data_df"]

    preprocessed = []
    
    for reactions, gt in arr:
        left, right = parse_reaction_smiles(reactions)
        prompt = prompt_templates[task_name].replace("[LEFT]", ", ".join(left)).replace("[RIGHT]", ", ".join(right))
        
        preprocessed.append({
            "query": prompt,
            "task": "yield_prediction",
            "subtask": task_name,
            "gt": gt,
            "meta": {
                "reactants": left,
                "products": right,
            }
        })
        
    with open(output_path, "w") as f:
        json.dump(preprocessed, f, indent=4)
        
def preprocess_yield_prediction(npz_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for task in task_names:
        preprocess_single_subtask(os.path.join(npz_dir, f"{task}_sample_100_test.npz"), os.path.join(output_dir, f"{task}.json"), task)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    preprocess_yield_prediction(args.npz_dir, args.output_dir)