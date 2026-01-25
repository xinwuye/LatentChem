import os, json
import sys
import logging
import os

# Import from local core for utils
from core.utils import extract_answer

# Import evaluator from parent eval's core directory
import importlib.util
import os

# Get the path to the parent eval's core directory
current_dir = os.path.dirname(os.path.abspath(__file__))
eval_core_dir = os.path.join(os.path.dirname(current_dir), "core")  # Changed from dirname(dirname(current_dir)) to dirname(current_dir)
evaluator_path = os.path.join(eval_core_dir, "evaluator.py")

evaluator_spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
evaluator_module = importlib.util.module_from_spec(evaluator_spec)
evaluator_spec.loader.exec_module(evaluator_module)

MoleculeSMILESEvaluator = evaluator_module.MoleculeSMILESEvaluator
evaluator = MoleculeSMILESEvaluator()
logger = logging.getLogger(__name__)

from core.task_evaluator import MolSimiliarityTaskEvaluator

def evaluate_molecule_design_score(model_name,  gt_path, logs_dir, results_dir,sample_count=1):
    mol_design_evaluator = MolSimiliarityTaskEvaluator()
    result = mol_design_evaluator.evaluate_score(model_name, sample_count, gt_path, logs_dir, "molecule_design")
    os.makedirs(os.path.join(results_dir, 'molecule_design'), exist_ok=True)
    with open(os.path.join(results_dir, 'molecule_design', f'{model_name}_{sample_count}.json'), 'w') as f:
        json.dump(result, f, indent=4)
    return result

def record_molecule_design_results(model_name, gt_path, logs_dir, results_dir, sample_count = 1):
    evaluator = MolSimiliarityTaskEvaluator()
        
    dataframe = evaluator.record_results(model_name, sample_count, gt_path, logs_dir, "molecule_design")
        
    os.makedirs(f"{results_dir}/molecule_design", exist_ok=True)
    dataframe.to_csv(f"{results_dir}/molecule_design/eval_results_{model_name}.csv", index=False)