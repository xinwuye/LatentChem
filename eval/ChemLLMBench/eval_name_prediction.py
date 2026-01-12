import sys
import logging
import json
import os

# Import from local core directory
local_core_dir = os.path.join(os.path.dirname(__file__), 'core')
if local_core_dir not in sys.path:
    sys.path.insert(0, local_core_dir)

from metrics import exact_match
from utils import extract_answer

logger = logging.getLogger(__name__)

def eval_name_prediction_from_list(pred_list, gt_list, task, total_length):
    correct_num = 0
    for i in range(len(pred_list)):
        pred = pred_list[i]
        gt = gt_list[i]
        if task == 'iupac2smiles':
            if exact_match(pred, gt):
                correct_num += 1
        if pred == gt:
            correct_num += 1
            
    return {
        "correct_rate": correct_num / total_length,
        f"{task}_valid_rate" : len(pred_list) / total_length
    }
    
def evaluate_name_prediction_score(model_name, gt_path, logs_dir, results_dir):
    result_dict = dict()
    
    for task in ['iupac2smiles', 'smiles2iupac']:
        logger.info(f'evaluating {task} for model {model_name}')
        file_name = f"{logs_dir}/{task}/{model_name}.json" 
        pred_results = json.load(open(file_name, "r"))
        
        gt_name = f"{gt_path}/{task}/{task}.json"
        gts = json.load(open(gt_name, "r"))
        
        invalid_number = 0
        pred_list, gt_list = list(), list()
        
        for i, pred in enumerate(pred_results):
            answer = extract_answer(pred['result'])
            if answer is None:
                invalid_number += 1
                continue
            pred_list.append(answer)
            gt = gts[i]
            meta = gt['meta']
            # Check if meta is already a dict or a JSON string
            if isinstance(meta, str):
                meta = json.loads(meta)
            gt_list.append(meta['reference'])
        
        assert len(gt_list) == len(pred_list)
        
        result_dict[task] = eval_name_prediction_from_list(pred_list, gt_list, task, len(pred_results))
    
    logger.info(f"eval_score_{model_name}_name_prediction:\n\r{result_dict}")
    os.makedirs(f"{results_dir}/name_prediction", exist_ok=True)
    json.dump(result_dict, open(f"{results_dir}/name_prediction/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_dict