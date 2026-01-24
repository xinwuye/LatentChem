import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from extract_molecules import extract_source_molecules
import pandas as pd
import argparse

def save_similarity_matrix_csv(sim_matrix, smiles_list, out_path):
    """
    将相似度矩阵写入 CSV，并以 smiles 作为行列名
    """
    df = pd.DataFrame(
        sim_matrix,
        index=smiles_list,
        columns=smiles_list
    )
    df.to_csv(out_path)

def load_molecule_embeddings_from_refined9(logp_dir):
    """
    每个文件 XXXXXX_refined_9.npy 表示一个分子
    文件内容是一个 list / array，包含所有 refined 层
    """
    pattern = re.compile(r"(\d+)_refined_9\.npy")

    mol_ids = []
    mol_embeddings = []

    # 1. 先找全局最大维度 z
    all_dims = []

    for fname in os.listdir(logp_dir):
        if not pattern.match(fname):
            continue

        layers = np.load(os.path.join(logp_dir, fname), allow_pickle=True)

        for emb in layers:
            emb = np.asarray(emb).reshape(-1)
            all_dims.append(emb.shape[0])

    z = max(all_dims)
    print(f"Using padded embedding dimension z = {z}")

    # 2. 再逐分子处理
    for fname in sorted(os.listdir(logp_dir)):
        match = pattern.match(fname)
        if match is None:
            continue

        mol_id = int(match.group(1))
        layers = np.load(os.path.join(logp_dir, fname), allow_pickle=True)

        padded_layers = []
        for emb in layers:
            emb = np.asarray(emb).reshape(-1)
            if emb.shape[0] < z:
                emb = np.pad(emb, (0, z - emb.shape[0]), mode="constant")
            padded_layers.append(emb)

        mol_emb = np.stack(padded_layers, axis=0).mean(axis=0)

        mol_ids.append(mol_id)
        mol_embeddings.append(mol_emb)

    return mol_ids, np.stack(mol_embeddings, axis=0)


def compute_similarity_matrix(mol_embeddings):
    return cosine_similarity(mol_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute similarity matrix from refined embeddings.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., logp)")
    parser.add_argument("--logp_dir", type=str, required=True, help="Directory containing refined_9 embeddings.")
    parser.add_argument("--json_file_path", type=str, required=True, help="Path to the JSON file with SMILES.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the similarity matrix CSV.")

    args = parser.parse_args()

    # Extract all source molecules
    smiles_list = extract_source_molecules(args.json_file_path)

    # Load molecule embeddings
    mol_ids, mol_embeddings = load_molecule_embeddings_from_refined9(args.logp_dir)

    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(mol_embeddings)

    # Save similarity matrix to CSV
    output_file = os.path.join(args.save_path, f"biotoken_similarity_matrix_refined9.csv")
    save_similarity_matrix_csv(sim_matrix, smiles_list, output_file)