def mol_to_canonical(smiles):
    """把 SMILES 转为 canonical SMILES；无法解析时返回原始字符串（保持一致性）。"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)
    except Exception:
        return smiles