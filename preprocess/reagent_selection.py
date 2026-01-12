import regex as re
import random
import pandas as pd
import json
import ast
import os
import argparse

task_names = ['ligand', 'reagent', 'solvent']

patterns = {
    'ligand': {
        "REACTANT1": r"reactant\s*1\s*:\s*([^\n\r]+)",
        "REACTANT2": r"reactant\s*2\s*:\s*([^\n\r]+)",
        "BASE": r"base\s*:\s*([^\n\r]+)",
        "SOLVENT": r"solvent\s*:\s*([^\n\r]+)",
    },
    'reagent': {
        "REACTANT": r"reactant\s*:\s*([^\n\r]+)",
        "LIGAND": r"ligand\s*:\s*([^\n\r]+)",
        "SOLVENT": r"solvent\s*:\s*([^\n\r]+)",
        "BASE": r"base\s*:\s*([^\n\r]+)",
    },
    'solvent': {
        "REACTANT1": r"reactant\s*1\s*:\s*([^\n\r]+)",
        "REACTANT2": r"reactant\s*2\s*:\s*([^\n\r]+)",
        "LIGAND": r"ligand\s*:\s*([^\n\r]+)",
        "BASE": r"base\s*:\s*([^\n\r]+)",
    }
}

prompt_templates = {
    'ligand': '''You are an expert chemist. Given selected two reactants, one reagent and solvent of a Suzuki reaction, predict the optimal ligand out of the given ones that maximize the yield with the rest of reaction components by using your experienced chemical ligand selection knowledge. Input: The SMILES of the reactants, reagent, and solvent. Output: The SMILES of the selected optimal ligand. Your final answer must be formatted as <answer> SMILES </answer>. Reactants: [REACTANT1], [REACTANT2]. Base: [BASE]. Solvent: [SOLVENT]. Ligand list for selection: [LIST]. ''',
    'reagent': '''You are an expert chemist. Given selected one reactant, two reagents and solvent of a Suzuki reaction, predict the optimal reactant out of the given ones that maximize the yield with the rest of reaction components by using your experienced chemical reactant selection knowledge. Input: The SMILES of the reactant, reagents, and solvent. Output: The SMILES of the selected optimal reactant. Your final answer must be formatted as <answer> SMILES </answer>. Reactants: [REACTANT]. Ligand: [LIGAND]. Base: [BASE]. Solvent: [SOLVENT]. Reactants list for selection: [LIST]. ''',
    'solvent': '''You are an expert chemist. Given selected two reactants, two reagents of a Suzuki reaction, predict the optimal solvent out of the given ones that maximize the yield with the rest of reaction components by using your experienced chemical solvent selection knowledge. Input: The SMILES of the reactants and reagents. Output: The SMILES of the selected optimal solvent. Your final answer must be formatted as <answer> SMILES </answer>. Reactants: [REACTANT1], [REACTANT2], Ligand: [LIGAND], Base: [BASE], Solvent list for selection: [LIST]'''
}

def extract_smiles(prompt, task_name):
    results = {}
    for key, pattern in patterns[task_name].items():
        matches = list(re.finditer(pattern, prompt, re.IGNORECASE))
        if matches:
            results[key] = matches[-1].group(1).strip()
            
    return results

def prepare_prompt(task_name, raw_prompt, smiles_list):
    smiles_list_str = ', '.join(smiles_list)
    inputs = extract_smiles(raw_prompt, task_name)
    prompt = prompt_templates[task_name].replace('[LIST]', smiles_list_str)
    for key, value in inputs.items():
        prompt = prompt.replace(f"[{key}]", value)
        
    return prompt, inputs

def preprocess_single_subtask(csv_path, output_path, task_name):
    df = pd.read_csv(csv_path)
    
    preprocessed = []
    
    random.seed(42)
    
    for i in range(len(df)):
        row = df.iloc[i]
        raw_prompt = row['task']
        candidate_rank = row['candidate_rank']
        smiles_list = ast.literal_eval(candidate_rank)
        random.shuffle(smiles_list)
        
        prompt, input_smiles = prepare_prompt(task_name, raw_prompt, smiles_list)
        preprocessed.append({
            "query": prompt,
            "task": "reagent_selection",
            "subtask": task_name,
            "gt": "",
            "meta": {
                "candidate_rank": candidate_rank,
                "reactants": list(input_smiles.values()),
                "reagents": smiles_list
            }
        })
        
    with open(output_path, "w") as f:
        json.dump(preprocessed, f, indent=4)
        
def preprocess_reagent_selection(csv_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for task in task_names:
        preprocess_single_subtask(os.path.join(csv_dir, f"{task}_sample.csv"), os.path.join(output_dir, f"{task}.json"), task)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    preprocess_reagent_selection(args.csv_dir, args.output_dir)
    