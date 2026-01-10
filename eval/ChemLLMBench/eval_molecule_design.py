import os, json
from core.utils import extract_answer
import logging
import os
from core.task_evaluator import MolSimiliarityTaskEvaluator

logger = logging.getLogger(__name__)

def evaluate_molecule_design_score(model_name, sample_count, gt_path, logs_dir, results_dir):
    mol_design_evaluator = MolSimiliarityTaskEvaluator()
    return mol_design_evaluator.run(model_name, sample_count, gt_path, logs_dir, results_dir, "molecule_design")