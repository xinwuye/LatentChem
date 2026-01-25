import sys
import logging
import json
import os

# Import from local core directory
local_core_dir = os.path.join(os.path.dirname(__file__), 'core')
if local_core_dir not in sys.path:
    sys.path.insert(0, local_core_dir)

from metrics import exact_match
from core.utils import extract_answer

logger = logging.getLogger(__name__)

from core.task_evaluator import MolSimiliarityTaskEvaluator

def evaluate_reaction_prediction_score(model_name,  gt_path, logs_dir, results_dir,sample_count=1):
    mol_design_evaluator = MolSimiliarityTaskEvaluator()
    result = mol_design_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, "reaction_prediction")
    os.makedirs(os.path.join(results_dir, 'reaction_prediction'), exist_ok=True)
    with open(os.path.join(results_dir, 'reaction_prediction', f'{model_name}_{sample_count}.json'), 'w') as f:
        json.dump(result, f, indent=4)
    return result

def record_reaction_prediction_results(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    evaluator = MolSimiliarityTaskEvaluator()
        
    dataframe = evaluator.record_results(model_name, sample_count, gt_path, logs_dir, "reaction_prediction")
        
    os.makedirs(f"{results_dir}/reaction_prediction", exist_ok=True)
    dataframe.to_csv(f"{results_dir}/reaction_prediction/eval_results_{model_name}.csv", index=False)

# def eval_reaction_prediction_from_list(pred_list, gt_list, task, total_length):
#     correct_num = 0
#     for i in range(len(pred_list)):
#         if exact_match(pred_list[i], gt_list[i]):
#             correct_num += 1
            
#     return {
#         "correct_rate": correct_num / total_length,
#         f"{task}_valid_rate" : len(pred_list) / total_length
#     }
    
# def evaluate_reaction_prediction_score(model_name, gt_path, logs_dir, results_dir):
#     result_dict = dict()
    
#     for task in ['reaction_prediction']:
#         logger.info(f'evaluating {task} for model {model_name}')
#         file_name = f"{logs_dir}/{task}/{model_name}.json" 
#         pred_results = json.load(open(file_name, "r"))
        
#         gt_name = f"{gt_path}/{task}/{task}.json"
#         gts = json.load(open(gt_name, "r"))
        
#         invalid_number = 0
#         pred_list, gt_list = list(), list()
        
#         for i, pred in enumerate(pred_results):
#             # Handle both 'result' and 'results' keys depending on the data format
#             if 'results' in pred:
#                 answer = extract_answer(pred['results'])
#             elif 'result' in pred:
#                 answer = extract_answer(pred['result'])
#             else:
#                 raise KeyError("Prediction must contain either 'result' or 'results' key")
#             if answer is None:
#                 invalid_number += 1
#                 continue
#             pred_list.append(answer)
#             gt = gts[i]
#             meta = gt['meta']
#             # Check if meta is already a dict or a JSON string
#             if isinstance(meta, str):
#                 meta = json.loads(meta)
#             gt_list.append(meta['reference'])
        
#         assert len(gt_list) == len(pred_list)
        
#         result_dict[task] = eval_reaction_prediction_from_list(pred_list, gt_list, task, len(pred_results))
    
#     logger.info(f"eval_score_{model_name}_reaction_prediction:\n\r{result_dict}")
#     os.makedirs(f"{results_dir}/reaction_prediction", exist_ok=True)
#     json.dump(result_dict, open(f"{results_dir}/reaction_prediction/eval_score_{model_name}.json", "w"), indent=4)
    
#     return result_dict