import os, json
from core.utils import extract_answer
import logging
import os
import sys
from core.evaluator import MoleculeCaptionEvaluator
evaluator = MoleculeCaptionEvaluator()
logger = logging.getLogger(__name__)

from core.task_evaluator import TextSimiliarityTaskEvaluator

def evaluate_molecule_captioning_score(model_name,  gt_path, logs_dir, results_dir,sample_count=1):
    mol_captioning_evaluator = TextSimiliarityTaskEvaluator()
    result = mol_captioning_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, "molecule_captioning")
    os.makedirs(os.path.join(results_dir, 'molecule_captioning'), exist_ok=True)
    with open(os.path.join(results_dir, 'molecule_captioning', f'{model_name}_{sample_count}.json'), 'w') as f:
        json.dump(result, f, indent=4)
    return result

def record_molecule_captioning_results(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    evaluator = TextSimiliarityTaskEvaluator()
        
    dataframe = evaluator.record_results(model_name, sample_count, gt_path, logs_dir, "molecule_captioning")
        
    os.makedirs(f"{results_dir}/molecule_captioning", exist_ok=True)
    dataframe.to_csv(f"{results_dir}/molecule_captioning/eval_results_{model_name}.csv", index=False)