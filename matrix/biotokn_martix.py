import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from extract_molecules import extract_source_molecules
import argparse


def generate_layerwise_similarity_matrices(
    refined_dir: str,
    smiles_list: list,
    save_dir: str,
    num_layers: int = 10,
    prefix: str = "biotoken"
):
    """
    对 refined_9.npy 中的每一层（1~num_layers）
    分别生成一个相似度矩阵并保存为 CSV
    """
    os.makedirs(save_dir, exist_ok=True)

    pattern = re.compile(r"(\d+)_refined_9\.npy")

    # layer_embeddings[i] = list of embeddings for layer i
    layer_embeddings = [[] for _ in range(num_layers)]
    mol_ids = []

    # 1. 收集每一层的 embedding
    for fname in sorted(os.listdir(refined_dir)):
        match = pattern.match(fname)
        if match is None:
            continue

        mol_id = int(match.group(1))
        layers = np.load(os.path.join(refined_dir, fname), allow_pickle=True)

        assert len(layers) >= num_layers, \
            f"{fname} has only {len(layers)} layers"

        for i in range(num_layers):
            emb = np.asarray(layers[i]).reshape(-1)
            layer_embeddings[i].append(emb)

        mol_ids.append(mol_id)

    num_mols = len(mol_ids)
    assert num_mols == len(smiles_list), \
        f"Mol count {num_mols} != SMILES count {len(smiles_list)}"

    # 2. 对每一层分别计算 similarity
    for i in range(num_layers):
        embs = layer_embeddings[i]

        # 2.1 padding 到该层的最大维度
        max_dim = max(e.shape[0] for e in embs)
        padded = []

        for e in embs:
            if e.shape[0] < max_dim:
                e = np.pad(e, (0, max_dim - e.shape[0]), mode="constant")
            padded.append(e)

        padded = np.stack(padded, axis=0)

        # 2.2 similarity
        sim_matrix = cosine_similarity(padded)

        # 2.3 保存
        df = pd.DataFrame(
            sim_matrix,
            index=smiles_list,
            columns=smiles_list
        )

        out_path = os.path.join(
            save_dir,
            f"{prefix}_similarity_layer{i+1}.csv"
        )
        df.to_csv(out_path)

        print(f"[Saved] Layer {i+1} → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate layer-wise similarity matrices.")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., logp)")
    parser.add_argument("--refined_dir", type=str, required=True, help="Directory containing refined_9 embeddings.")
    parser.add_argument("--json_file_path", type=str, required=True, help="Path to the JSON file with SMILES.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the similarity matrices.")
    parser.add_argument("--num_layers", type=int, default=10, help="Number of layers to process.")
    parser.add_argument("--prefix", type=str, default="biotoken", help="Prefix for output files.")

    args = parser.parse_args()

    # Extract all source molecules
    smiles_list = extract_source_molecules(args.json_file_path)

    # Generate layer-wise similarity matrices
    generate_layerwise_similarity_matrices(
        refined_dir=args.refined_dir,
        smiles_list=smiles_list,
        save_dir=args.save_dir,
        num_layers=args.num_layers,
        prefix=args.prefix
    )