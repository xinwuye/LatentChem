
import json
import re
# from extract_molecules import extract_source_molecules
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
def extract_final_target_molecule(text: str):
    """
    Extract 'Final Target Molecule' SMILES from a raw string.
    Returns the SMILES string if found, otherwise None.
    """
    patterns = [
        # 处理 \"Final Target Molecule\": \"SMILES\"
        r'Final Target Molecule\\":\\s*\\"([^"]+)\\"',
        # 处理 "Final Target Molecule": "SMILES"
        r'"Final Target Molecule"\s*:\s*"([^"]+)"'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return None

path = "/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/outputs/124-taskthinker-bioupdater_matrix/results/124-taskthinker.json"

def load_concatenated_json(path):
    with open(path, "r") as f:
        text = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    objs = []

    while idx < len(text):
        text = text.lstrip()
        try:
            obj, offset = decoder.raw_decode(text)
            objs.append(obj)
            text = text[offset:]
        except json.JSONDecodeError:
            break

    return objs

# ---------- 主逻辑开始 ----------
def get_all_final_target_molecules(path,reference_path):
   
    data = load_concatenated_json(path)

    # 给定的 smiles 列表（按这个顺序输出）
    # 2. 分子 SMILES 列表
    json_file_path = reference_path
    # Extract all source molecules
    smiles_list = extract_source_molecules(json_file_path)

    # 建立 smiles -> Final Target Molecule 的映射
    smiles_to_result = {}

    for item in data:
        for tr in item["test_results"]:
            smiles = tr.get("smiles")[0]
            print(smiles)
            if smiles is None:
                continue
            smiles_to_result[smiles] = extract_final_target_molecule(tr["result"])

    # 按给定 smiles 列表顺序生成结果
    all_results = [
        smiles_to_result.get(smiles)  # 不存在则返回 None
        for smiles in smiles_list
    ]
    return all_results



    # 输出
    # print(all_results)
    # print(len(all_results))
