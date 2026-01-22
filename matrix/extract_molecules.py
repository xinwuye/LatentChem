import json
import re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
import pandas as pd
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
        # Extract the query field which contains the Source Molecule
        query = entry.get("query", "")
        
        # Use regex to find the Source Molecule part
        # Pattern looks for "Source Molecule:" followed by the SMILES string
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
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fingerprints.append(fp)
        else:
            raise ValueError(f"Invalid SMILES: {smiles}")
    return fingerprints
def calculate_similarity_matrix(fingerprints,molecules):
    """
    Calculate the similarity matrix using Tanimoto similarity.
    
    Args:
        fingerprints (list): List of RDKit fingerprint objects.
    
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
    index=molecules,   # 行标签：SMILES
    columns=molecules  # 列标签：SMILES
    )

# 5. 写入 CSV 文件
    output_file = "molecular_similarity_matrix_new.csv"
    df.to_csv(output_file, float_format="%.4f")
    print(f"相似度矩阵已保存到: {output_file}")
    
    return similarity_matrix
def main():
    # Specify the path to your JSON file
    json_file_path = "/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/data/ChemCoTBench/chemcotbench/mol_opt/qed.json"
    
    # Extract all source molecules
    molecules = extract_source_molecules(json_file_path)
    #get fignerprints
    fingerprints = generate_fingerprints(molecules)
    
    similarity_matrix = calculate_similarity_matrix(fingerprints,molecules)
    print("Similarity Matrix:")
    print(similarity_matrix)
    
    # # Print all extracted molecules
    # print(f"Found {len(molecules)} source molecules:")
    # for i, mol in enumerate(molecules, 1):
    #     print(f"{i}: {mol}")
    
    # # Optionally, save to a text file
    # output_file = "extracted_molecules.txt"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for mol in molecules:
    #         f.write(f"{mol}\n")
    
    # print(f"All molecules saved to {output_file}")

if __name__ == "__main__":
    main()
