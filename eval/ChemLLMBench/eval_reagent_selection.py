import sys
import logging
import json
import os

# Import from local core directory
local_core_dir = os.path.join(os.path.dirname(__file__), 'core')
if local_core_dir not in sys.path:
    sys.path.insert(0, local_core_dir)

from utils import extract_answer
from metrics import try_canonicalize_smiles

logger = logging.getLogger(__name__)

def eval_topk(ranked_list, pred, topk = 1):
    top_list = ranked_list[:topk]
    top_list = [try_canonicalize_smiles(x) for x in top_list]
    pred = try_canonicalize_smiles(pred)
    if pred is None: return False
    return pred in top_list

def eval_reagent_selection_from_list(pred_list, gt_list, task, top_k, total_length):
    correct_num = 0
    for i in range(len(pred_list)):
        pred = pred_list[i]
        gt = gt_list[i]
        if eval_topk(gt, pred, top_k):
            correct_num += 1
            
    return {
        "correct_rate": correct_num / total_length,
        f"{task}_valid_rate" : len(pred_list) / total_length
    }
    
def evaluate_reagent_selection_score(model_name, gt_path, logs_dir, results_dir):
    result_dict = dict()
    
    topk_dict = {
        "ligand": 5,
        "reactant": 1,
        "solvent": 1
    }
    
    for task in topk_dict.keys():
        logger.info(f'evaluating {task} for model {model_name}')
        file_name = f"{logs_dir}/{task}/{model_name}.json" 
        pred_results = json.load(open(file_name, "r"))
        
        gt_name = f"{gt_path}/{task}/{task}.json"
        gts = json.load(open(gt_name, "r"))
        
        invalid_number = 0
        pred_list, gt_list = list(), list()
        
        for i, pred in enumerate(pred_results):
            answer = extract_answer(pred['results'])
            if answer is None:
                invalid_number += 1
                continue
            pred_list.append(answer)
            gt = gts[i]
            meta = gt['meta']
            # Check if meta is already a dict or a JSON string
            if isinstance(meta, str):
                meta = json.loads(meta)
            gt_list.append(meta['candidate_rank'])
        
        assert len(gt_list) == len(pred_list)
        
        result_dict[task] = eval_reagent_selection_from_list(pred_list, gt_list, task, topk_dict[task], len(pred_results))
    
    logger.info(f"eval_score_{model_name}_reagent_selection:\n\r{result_dict}")
    os.makedirs(f"{results_dir}/reagent_selection", exist_ok=True)
    json.dump(result_dict, open(f"{results_dir}/reagent_selection/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_dict