
from ChemLLMBench.core.metrics import exact_match
from core.utils import extract_answer
import logging
import json
import os

logger = logging.getLogger(__name__)

from core.task_evaluator import MolSimiliarityTaskEvaluator

def eval_retro_from_list(pred_list, gt_list, task, total_length):
    correct_num = 0
    for i in range(len(pred_list)):
        if exact_match(pred_list[i], gt_list[i]):
            correct_num += 1
            
    return {
        "correct_rate": correct_num / total_length,
        f"{task}_valid_rate" : len(pred_list) / total_length
    }

def evaluate_retro_score(model_name,  gt_path, logs_dir, results_dir, sample_count=1):
    mol_design_evaluator = MolSimiliarityTaskEvaluator()
    result = mol_design_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, "retro")
    os.makedirs(os.path.join(results_dir, 'retro'), exist_ok=True)
    with open(os.path.join(results_dir, 'retro', f'{model_name}_{sample_count}.json'), 'w') as f:
        json.dump(result, f, indent=4)
    return result

def record_retro_results(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    evaluator = MolSimiliarityTaskEvaluator()
        
    dataframe = evaluator.record_results(model_name, sample_count, gt_path, logs_dir, "retro")
        
    os.makedirs(f"{results_dir}/retro", exist_ok=True)
    dataframe.to_csv(f"{results_dir}/retro/eval_results_{model_name}.csv", index=False)
