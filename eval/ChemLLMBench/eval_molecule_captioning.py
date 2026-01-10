import os, json
from core.utils import extract_answer
import logging
import os
import statistics

from core.task_evaluator import TextSimiliarityTaskEvaluator

def evaluate_molecule_captioning_score(model_name, sample_count, gt_path, logs_dir, results_dir):
    mol_captioning_evaluator = TextSimiliarityTaskEvaluator()
    return mol_captioning_evaluator.run(model_name, sample_count, gt_path, logs_dir, results_dir, "molecule_captioning")