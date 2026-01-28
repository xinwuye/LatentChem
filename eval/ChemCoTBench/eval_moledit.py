import json
from tqdm import tqdm
import os

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
# from rdkit.Chem import Draw
from rdkit.Chem.Fingerprints import FingerprintMols
import logging

from core.utils import extract_answer

logger = logging.getLogger(__name__)

def mol_prop(mol, prop):
    try:
        mol = Chem.MolFromSmiles(mol)
    except:
        return None
    # always remember to check if mol is None
    if mol is None:
        return None
    
    ## Basic Properties
    if prop == 'logP':
        return Descriptors.MolLogP(mol)
    elif prop == 'weight':
        return Descriptors.MolWt(mol)
    elif prop == 'qed':
        return Descriptors.qed(mol)
    elif prop == 'TPSA':
        return Descriptors.TPSA(mol)
    elif prop == 'HBA': # Hydrogen Bond Acceptor
        return Descriptors.NumHAcceptors(mol)
    elif prop == 'HBD': # Hydrogen Bond Donor
        return Descriptors.NumHDonors(mol)
    elif prop == 'rot_bonds': # rotatable bonds
        return Descriptors.NumRotatableBonds(mol)
    elif prop == 'ring_count':
        return Descriptors.RingCount(mol)
    elif prop == 'mr': # Molar Refractivity
        return Descriptors.MolMR(mol)
    elif prop == 'balabanJ':
        return Descriptors.BalabanJ(mol)
    elif prop == 'hall_kier_alpha':
        return Descriptors.HallKierAlpha(mol)
    elif prop == 'logD':
        return Descriptors.MolLogP(mol)
    elif prop == 'MR':
        return Descriptors.MolMR(mol)

    ## If Molecule is valid
    elif prop == 'validity':   
        # print(mol)
        return True
    
    ## Bond Counts
    elif prop == 'num_single_bonds':
        return sum([bond.GetBondType() == Chem.rdchem.BondType.SINGLE for bond in mol.GetBonds()])
    elif prop == 'num_double_bonds':
        return sum([bond.GetBondType() == Chem.rdchem.BondType.DOUBLE for bond in mol.GetBonds()])
    elif prop == 'num_triple_bonds':
        return sum([bond.GetBondType() == Chem.rdchem.BondType.TRIPLE for bond in mol.GetBonds()])
    elif prop == 'num_aromatic_bonds':
        return sum([bond.GetBondType() == Chem.rdchem.BondType.AROMATIC for bond in mol.GetBonds()])
    elif prop == 'num_rotatable_bonds': # rotatable bonds
        return Descriptors.NumRotatableBonds(mol)

    
    ## Common Atom Counts
    elif prop == 'num_carbon':
        return sum([atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()])
    elif prop == 'num_nitrogen':
        return sum([atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()])
    elif prop == 'num_oxygen':
        return sum([atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()])
    elif prop == 'num_fluorine':
        return sum([atom.GetAtomicNum() == 9 for atom in mol.GetAtoms()])
    elif prop == 'num_phosphorus':
        return sum([atom.GetAtomicNum() == 15 for atom in mol.GetAtoms()])
    elif prop == 'num_sulfur':
        return sum([atom.GetAtomicNum() == 16 for atom in mol.GetAtoms()])
    elif prop == 'num_chlorine':
        return sum([atom.GetAtomicNum() == 17 for atom in mol.GetAtoms()])
    elif prop == 'num_bromine':
        return sum([atom.GetAtomicNum() == 35 for atom in mol.GetAtoms()])
    elif prop == 'num_iodine':
        return sum([atom.GetAtomicNum() == 53 for atom in mol.GetAtoms()])
    elif prop == "num_boron":
        return sum([atom.GetAtomicNum() == 5 for atom in mol.GetAtoms()])
    elif prop == "num_silicon":
        return sum([atom.GetAtomicNum() == 14 for atom in mol.GetAtoms()])
    elif prop == "num_selenium":
        return sum([atom.GetAtomicNum() == 34 for atom in mol.GetAtoms()])
    elif prop == "num_tellurium":
        return sum([atom.GetAtomicNum() == 52 for atom in mol.GetAtoms()])
    elif prop == "num_arsenic":
        return sum([atom.GetAtomicNum() == 33 for atom in mol.GetAtoms()])
    elif prop == "num_antimony":
        return sum([atom.GetAtomicNum() == 51 for atom in mol.GetAtoms()])
    elif prop == "num_bismuth":
        return sum([atom.GetAtomicNum() == 83 for atom in mol.GetAtoms()])
    elif prop == "num_polonium":
        return sum([atom.GetAtomicNum() == 84 for atom in mol.GetAtoms()])
    
    ## Functional groups
    elif prop == "num_benzene":
        smarts = '[cR1]1[cR1][cR1][cR1][cR1][cR1]1'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_benzene_ring":
        smarts = '[cR1]1[cR1][cR1][cR1][cR1][cR1]1'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_hydroxyl":
        smarts = '[OX2H]'   # Hydroxyl including phenol, alcohol, and carboxylic acid.
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_anhydride":
        smarts = '[CX3](=[OX1])[OX2][CX3](=[OX1])'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_aldehyde":
        smarts = '[CX3H1](=O)[#6]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_ketone":
        smarts = '[#6][CX3](=O)[#6]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_carboxyl":
        smarts = '[CX3](=O)[OX2H1]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_ester":
        smarts = '[#6][CX3](=O)[OX2H0][#6]'    # Ester Also hits anhydrides but won't hit formic anhydride.
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_amide":
        smarts = '[NX3][CX3](=[OX1])[#6]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_amine":
        smarts = '[NX3;H2,H1;!$(NC=O)]'    # Primary or secondary amine, not amide.
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_nitro":
        smarts = '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_halo":
        smarts = '[F,Cl,Br,I]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_thioether":
        smarts = '[SX2][CX4]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_nitrile":
        smarts = '[NX1]#[CX2]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_thiol":
        smarts = '[#16X2H]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_sulfide":
        smarts = '[#16X2H0]'    #  Won't hit thiols. Hits disulfides too.
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        exception = '[#16X2H0][#16X2H0]'
        matches_exception = mol.GetSubstructMatches(Chem.MolFromSmarts(exception))
        return len(matches) - len(matches_exception)
    elif prop == "num_disulfide":
        smarts = '[#16X2H0][#16X2H0]'    
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_sulfoxide":
        smarts = '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_sulfone":
        smarts = '[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)
    elif prop == "num_borane":
        smarts = '[BX3]'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
        return len(matches)

    else:
        raise ValueError(f'Property {prop} not supported')

def is_valid_smiles(smiles):
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except:
        return False

GROUP_SET={
    "benzene",
    "benzene_ring",
    "hydroxyl",
    "anhydride",
    "aldehyde",
    "ketone",
    "carboxyl",
    "ester",
    "amide",
    "amine",
    "nitro",
    "halo",
    "thioether",
    "nitrile",
    "thiol",
    "sulfide",
    "disulfide",
    "sulfoxide",
    "sulfone",
    "borane",    
}

GROUP_TO_SMARTS = {
    # 芳香环类
    "benzene": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",       # 苯环(使用环原子标记)
    "benzene_ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",  # 苯环(使用环原子标记)
    
    # 含氧官能团
    "hydroxyl": "[OX2H]",                                # 羟基(包括酚羟基和醇羟基)
    "aldehyde": "[CX3H1](=O)[#6]",                       # 醛基
    "ketone": "[#6][CX3](=O)[#6]",                       # 酮基
    "carboxyl": "[CX3](=O)[OX2H1]",                      # 羧基
    "ester": "[#6][CX3](=O)[OX2H0][#6]",                 # 酯基(排除酸酐)
    "anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",      # 酸酐
    
    # 含氮官能团
    "amine": "[NX3;H2,H1;!$(NC=O)]",                     # 伯胺或仲胺(排除酰胺)
    "amide": "[NX3][CX3](=[OX1])[#6]",                   # 酰胺
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",  # 硝基
    
    # 卤素
    "halo": "[F,Cl,Br,I]",                               # 卤素
    
    # 含硫官能团
    "thiol": "[#16X2H]",                                 # 巯基
    "thioether": "[SX2][CX4]",                           # 硫醚
    "disulfide": "[#16X2H0][#16X2H0]",                   # 二硫键
    "sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]", # 亚砜
    "sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",  # 砜
    "sulfide": "[#16X2H0]",                              # 硫醚(排除二硫键)
    
    # 其他
    "nitrile": "[NX1]#[CX2]",                            # 氰基
    "borane": "[BX3]",                                   # 硼烷基
}

def check_edit_add_valid(src, tgt, group)->bool:
    if group not in GROUP_SET:
        logger.warning(f"Unknown group: {group}")
    assert group in GROUP_SET
    assert is_valid_smiles(src), f"无效的源分子SMILES: {src}" 
    try:
        assert is_valid_smiles(tgt), f"无效的目标分子SMILES: {tgt}"
    except Exception as e:
        logger.debug(e)
        return False
    if mol_prop(tgt, "num_"+group) == mol_prop(src, "num_"+group) + 1:
        return True
    else:
        logger.debug(f"添加{group}失败: 目标分子中{group}数量为{mol_prop(tgt, 'num_' + group)}, 源分子中{group}数量为{mol_prop(src, 'num_' + group)}")
        return False

def check_edit_del_valid(src, tgt, group)->bool:
    assert group in GROUP_SET
    assert is_valid_smiles(src), f"无效的源分子SMILES: {src}" 
    try:
        assert is_valid_smiles(tgt), f"无效的目标分子SMILES: {tgt}"
    except Exception as e:
        logger.debug(e)
        return False
    return mol_prop(tgt, "num_"+group) == mol_prop(src, "num_"+group) - 1

def check_edit_sub_valid(src, tgt, remove_group, add_group)->bool:
    assert remove_group in GROUP_SET
    assert add_group in GROUP_SET
    assert is_valid_smiles(src), f"无效的源分子SMILES: {src}" 
    try:
        assert is_valid_smiles(tgt), f"无效的目标分子SMILES: {tgt}"
    except Exception as e:
        logger.debug(e)
        return False
    return mol_prop(tgt, "num_"+remove_group) == mol_prop(src, "num_"+remove_group) - 1 and mol_prop(tgt, "num_"+add_group) == mol_prop(src, "num_"+add_group) + 1

def calculate_molecular_similarity(mol1, mol2, fingerprint_type='Morgan', 
                                 radius=2, 
                                 n_bits=2048,
                                 similarity_metric='Tanimoto'):
    """
    计算两个分子之间的相似度
    
    参数:
    - mol1, mol2: RDKit分子对象或SMILES字符串
    - fingerprint_type: 指纹类型，可选 'Morgan', 'RDKit', 'AtomPairs', 'TopologicalTorsion', 'MACCS'
    - radius: Morgan指纹的半径（仅对Morgan指纹有效）
    - n_bits: 指纹的位数（对Morgan和RDKit指纹有效）
    - similarity_metric: 相似度度量方法，可选 'Tanimoto', 'Dice', 'Cosine', 'Sokal', 'Russel'等
    
    返回:
    - 相似度分数 (0-1之间)
    """
    
    # 如果输入是SMILES字符串，先转换为分子对象
    if isinstance(mol1, str):
        mol1 = Chem.MolFromSmiles(mol1)
    if isinstance(mol2, str):
        mol2 = Chem.MolFromSmiles(mol2)
    
    if mol1 is None or mol2 is None:
        # raise ValueError("无效的分子输入")
        return 0.0
    
    # 生成指纹
    if fingerprint_type == 'Morgan':
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=radius, nBits=n_bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=radius, nBits=n_bits)
    elif fingerprint_type == 'RDKit':
        fp1 = FingerprintMols.FingerprintMol(mol1, minPath=1, maxPath=7, fpSize=n_bits)
        fp2 = FingerprintMols.FingerprintMol(mol2, minPath=1, maxPath=7, fpSize=n_bits)
    elif fingerprint_type == 'AtomPairs':
        fp1 = Chem.rdMolDescriptors.GetAtomPairFingerprint(mol1)
        fp2 = Chem.rdMolDescriptors.GetAtomPairFingerprint(mol2)
    elif fingerprint_type == 'TopologicalTorsion':
        fp1 = Chem.rdMolDescriptors.GetTopologicalTorsionFingerprint(mol1)
        fp2 = Chem.rdMolDescriptors.GetTopologicalTorsionFingerprint(mol2)
    elif fingerprint_type == 'MACCS':
        fp1 = AllChem.GetMACCSKeysFingerprint(mol1)
        fp2 = AllChem.GetMACCSKeysFingerprint(mol2)
    else:
        # raise ValueError(f"不支持的指纹类型: {fingerprint_type}")
        return 0.0
    
    # 计算相似度
    if fingerprint_type in ['Morgan', 'RDKit', 'MACCS']:
        if similarity_metric == 'Tanimoto':
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif similarity_metric == 'Dice':
            return DataStructs.DiceSimilarity(fp1, fp2)
        elif similarity_metric == 'Cosine':
            return DataStructs.CosineSimilarity(fp1, fp2)
        else:
            # raise ValueError(f"不支持的相似度度量方法: {similarity_metric}")
            return 0.0
    else:  # 对于AtomPairs和TopologicalTorsion指纹
        if similarity_metric == 'Tanimoto':
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        else:
            raise ValueError(f"对于{fingerprint_type}指纹，只支持Tanimoto相似度")

from core.task_evaluator import BaseTaskEvaluator
class MolEditEvaluator(BaseTaskEvaluator):
    def extract_gt(self, gt_raw_item, task):
        '''
        GT is not used for this task, all use meta
        '''
        return ""
    def prepare_metadata(self, sample):
        meta = json.loads(sample['meta'])
        return meta
    
    def evaluate_predictions(self, preds, gts, total_len, metadata = None, task_name = None):
        correct_num = 0
        if len(preds) == 0:
            return{
                "correct_rate": 0,
                "valid-rate": 0
            }
        for i in range(len(preds)):
            if task_name in ['add']:
                if check_edit_add_valid(src=metadata[i]['molecule'], tgt=preds[i], group=metadata[i]['added_group']):
                    correct_num += 1
            if task_name in ['delete']:
                if check_edit_del_valid(src=metadata[i]['molecule'], tgt=preds[i], group=metadata[i]['removed_group']):
                    correct_num += 1
            if task_name == 'sub':
                if check_edit_sub_valid(src=metadata[i]['molecule'], tgt=preds[i], remove_group=metadata[i]['removed_group'], add_group=metadata[i]['added_group']):
                    correct_num += 1
        
        my_dict = {
            "correct_rate": correct_num / total_len,
            # f"{task}-valid-rate": len(preds) / total_len,
        }
        return my_dict
    
def evaluate_moledit_score(model_name, gt_path, logs_dir, results_dir, sample_count = 1): 
    result_dict = dict()
    evaluator = MolEditEvaluator()
    
    for task in ['add', 'delete', 'sub']:
        logger.info(f'evaluating {task} for model {model_name}')
        
        result_dict[task] = evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, task)
    
    logger.info(f"eval_score_{model_name}_moledit:\n\r{result_dict}")
    os.makedirs(f"{results_dir}/moledit", exist_ok=True)
    json.dump(result_dict, open(f"{results_dir}/moledit/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_dict

def record_moledit_results(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    evaluator = MolEditEvaluator()
    for task in ['add', 'delete', 'sub']:
        logger.info(f'recording {task} for model {model_name}')
        
        dataframe = evaluator.record_results(model_name, sample_count, gt_path, logs_dir, task)
        
        os.makedirs(f"{results_dir}/moledit/{task}", exist_ok=True)
        dataframe.to_csv(f"{results_dir}/moledit/{task}/eval_results_{model_name}.csv", index=False)