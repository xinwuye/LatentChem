#!/usr/bin/env python
"""
Wrapper script to run ChemLLMBench evaluation from the parent eval directory
to properly handle module imports.
"""

import os
import sys

# Add the eval directory to the Python path
eval_dir = os.path.dirname(os.path.abspath(__file__))
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

# Import from ChemLLMBench using absolute imports
from ChemLLMBench.eval_molecule_captioning import evaluate_molecule_captioning_score
from ChemLLMBench.eval_molecule_design import evaluate_molecule_design_score
from ChemLLMBench.eval_name_prediction import evaluate_name_prediction_score
from ChemLLMBench.eval_property_prediction import evaluate_property_prediction_score
from ChemLLMBench.eval_reaction_prediction import evaluate_reaction_prediction_score
from ChemLLMBench.eval_reagent_selection import evaluate_reagent_selection_score
from ChemLLMBench.eval_retro import evaluate_retro_score
from ChemLLMBench.eval_yield_prediction import evaluate_yield_prediction_score

def eval_all_ChemLLMBench(log_name, dataset_path, logs_dir, results_dir, sample_count=None):
    evaluate_molecule_captioning_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_molecule_design_score(log_name, dataset_path, logs_dir, results_dir)
    # evaluate_name_prediction_score(log_name, dataset_path, logs_dir, results_dir)
    # evaluate_property_prediction_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_reaction_prediction_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_reagent_selection_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_retro_score(log_name, dataset_path, logs_dir, results_dir)
    # evaluate_yield_prediction_score(log_name, dataset_path, logs_dir, results_dir)


if __name__ == "__main__":
    eval_all_ChemLLMBench(
        "inference_result_0_7_no",
        "/BioLatent/Bio-LatentCOT/data/ChemLLMBench/chemllmbench",
        "/BioLatent/Bio-LatentCOT/data/chemllmbench",
        "/BioLatent/Bio-LatentCOT/eval/results"
    )