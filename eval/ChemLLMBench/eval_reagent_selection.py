import sys
import logging
import json
import os

# Import from local core directory
local_core_dir = os.path.join(os.path.dirname(__file__), 'core')
if local_core_dir not in sys.path:
    sys.path.insert(0, local_core_dir)

from core.utils import extract_answer
from core.task_evaluator import BaseTaskEvaluator
from metrics import try_canonicalize_smiles

logger = logging.getLogger(__name__)

topk_dict = {
    "ligand": 5,
    "reactant": 1,
    "solvent": 1
}

def eval_topk(ranked_list, pred, topk = 1):
    top_list = ranked_list[:topk]
    print(type(ranked_list))
    top_list = [try_canonicalize_smiles(x) for x in top_list]
    pred = try_canonicalize_smiles(pred)
    if pred is None: return False
    return pred in top_list

def eval_reagent_selection_from_list(pred_list, gt_list, task, top_k, total_length):
    correct_num = 0
    for i in range(len(pred_list)):
        pred = pred_list[i]
        gt = gt_list[i]
        if eval_topk(eval(gt), pred, top_k):
            correct_num += 1
            
    return {
        "correct_rate": correct_num / total_length,
        f"{task}_valid_rate" : len(pred_list) / total_length
    }

class ReagentSelectionEvaluator(BaseTaskEvaluator):
    def extract_gt(self, gt_raw_item, task_name):
        meta = gt_raw_item['meta']
        if isinstance(meta, str):
            meta = json.loads(meta)
        candidate_rank = meta['candidate_rank']
        # print(candidate_rank)
        if isinstance(candidate_rank, str):
            candidate_rank =candidate_rank
        return candidate_rank
    
    def prepare_metadata(self, sample):
        return None
    
    def evaluate_predictions(self, preds, gts, total_len, metadata, task_name):
        return eval_reagent_selection_from_list(preds, gts, task_name, topk_dict[task_name], total_len)

def evaluate_reagent_selection_score(model_name, gt_path, logs_dir, results_dir, sample_count = 1): 
    result_dict = dict()
    evaluator = ReagentSelectionEvaluator()
    
    for task in topk_dict.keys():
        logger.info(f'evaluating {task} for model {model_name}')
        
        result_dict[task] = evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, task)
    
    logger.info(f"eval_score_{model_name}_reagent_selection:\n\r{result_dict}")
    os.makedirs(f"{results_dir}/reagent_selection", exist_ok=True)
    json.dump(result_dict, open(f"{results_dir}/reagent_selection/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_dict

def record_reagent_selection_results(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    evaluator = ReagentSelectionEvaluator()
    for task in topk_dict.keys():
        logger.info(f'recording {task} for model {model_name}')
        
        dataframe = evaluator.record_results(model_name, sample_count, gt_path, logs_dir, task)
        
        os.makedirs(f"{results_dir}/reagent_selection/{task}", exist_ok=True)
        dataframe.to_csv(f"{results_dir}/reagent_selection/{task}/eval_results_{model_name}.csv", index=False)