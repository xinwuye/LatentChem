

import json
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
import pandas as pd
from answer_result_and_match import get_all_final_target_molecules
import argparse

def extract_source_molecules(json_file_path):
    """
    Extract all Source Molecules from the qed.json file.
    
    Args:
        json_file_path (str): Path to the JSON file
    
    Returns:
        list: A list of SMILES strings representing the source molecules
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    molecules = []
    
    for entry in data:
        query = entry.get("query", "")
        match = re.search(r"Source Molecule:\s*([^\n]+)", query)
        
        if match:
            smiles = match.group(1).strip()
            smiles = smiles.replace('.', '')
            molecules.append(smiles)
    
    return molecules

def generate_fingerprints(smiles_list, radius=2, n_bits=1024):
    """
    Generate Morgan fingerprints for a list of molecules.
    
    Args:
        smiles_list (list): List of SMILES strings.
        radius (int): Radius for Morgan fingerprint.
        n_bits (int): Number of bits in the fingerprint.
    
    Returns:
        list: List of RDKit fingerprint objects.
    """
    fingerprints = []
    for smiles in smiles_list:
        if smiles is None or smiles.strip() == "":
            smiles = "C"  # Use methane as a placeholder for invalid SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(fp)
        else:
            mol = Chem.MolFromSmiles("C")  # Use methane as a placeholder for invalid SMILES
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(fp)
    return fingerprints

def calculate_similarity_matrix(fingerprints, molecules, output_file):
    """
    Calculate the similarity matrix using Tanimoto similarity.
    
    Args:
        fingerprints (list): List of RDKit fingerprint objects.
        molecules (list): List of SMILES strings.
        output_file (str): Path to save the similarity matrix CSV file.
    
    Returns:
        np.ndarray: Similarity matrix.
    """
    num_molecules = len(fingerprints)
    similarity_matrix = np.zeros((num_molecules, num_molecules))
    
    for i in range(num_molecules):
        for j in range(i, num_molecules):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix
    
    df = pd.DataFrame(
        similarity_matrix,
        index=molecules,   # Row labels: SMILES
        columns=molecules  # Column labels: SMILES
    )
    
    df.to_csv(output_file, float_format="%.4f")
    print(f"Similarity matrix saved to: {output_file}")
    
    return similarity_matrix

def main():
    parser = argparse.ArgumentParser(description="Calculate molecular similarity matrix.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the similarity matrix CSV file.")
    parser.add_argument("--answer_file", type=str, required=True, help="Path to the answer JSON file.")
    parser.add_argument("--reference_path_file", type=str, required=True, help="Path to the answer JSON file.")
    
    args = parser.parse_args()
    
    # Get all final target molecules from the answer file
    molecules = get_all_final_target_molecules(args.answer_file,args.reference_path_file)
    
    # Generate fingerprints
    fingerprints = generate_fingerprints(molecules)
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(fingerprints, molecules, args.output_file)
    print("Similarity Matrix:")
    print(similarity_matrix)

if __name__ == "__main__":
    main()