import sys
import logging
import json
import os

# Add the parent directory to the path to access the main core module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Changed from dirname(dirname(current_dir)) to dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from core.utils import extract_answer

logger = logging.getLogger(__name__)

def eval_property_prediction_from_list(pred_list, gt_list, task, total_length):
    correct_num = 0
    for i in range(len(pred_list)):
        pred = pred_list[i]
        gt = gt_list[i]
        if str(pred).lower() == str(gt).lower():
            correct_num += 1
            
    return {
        "correct_rate": correct_num / total_length,
        f"{task}_valid_rate" : len(pred_list) / total_length
    }
    
def evaluate_property_prediction_score(model_name, gt_path, logs_dir, results_dir):
    result_dict = dict()
    
    for task in ["BACE", "BBBP", "ClinTox", "HIV", "Tox"]:
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
            gt_list.append(gt['gt'])
        
        assert len(gt_list) == len(pred_list)
        
        result_dict[task] = eval_property_prediction_from_list(pred_list, gt_list, task, len(pred_results))
    
    logger.info(f"eval_score_{model_name}_property_prediction:\n\r{result_dict}")
    os.makedirs(f"{results_dir}/property_prediction", exist_ok=True)
    json.dump(result_dict, open(f"{results_dir}/property_prediction/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_dict