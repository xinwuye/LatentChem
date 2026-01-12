from rdkit import Chem
import regex as re

def try_canonicalize_smiles(smiles):
    """如果能被 RDKit 解析则返回 canonical SMILES，否则返回 None"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None
    
def exact_match(smiles1, smiles2):
    canonicalized1 = try_canonicalize_smiles(smiles1)
    canonicalized2 = try_canonicalize_smiles(smiles2)
    return canonicalized1 == canonicalized2 and canonicalized1 is not None