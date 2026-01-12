import os
import sys
# sys.path.append("/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval")
from ChemLLMBench.eval_molecule_captioning import evaluate_molecule_captioning_score
from ChemLLMBench.eval_molecule_design import evaluate_molecule_design_score
from ChemLLMBench.eval_name_prediction import evaluate_name_prediction_score
from ChemLLMBench.eval_property_prediction import evaluate_property_prediction_score
from ChemLLMBench.eval_reaction_prediction import evaluate_reaction_prediction_score
from ChemLLMBench.eval_reagent_selection import evaluate_reagent_selection_score
from ChemLLMBench.eval_retro import evaluate_retro_score
from ChemLLMBench.eval_yield_prediction import evaluate_yield_prediction_score

def eval_all_ChemLLMBench(log_name, dataset_path, logs_dir, results_dir):
    evaluate_molecule_captioning_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_molecule_design_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_name_prediction_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_property_prediction_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_reaction_prediction_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_reagent_selection_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_retro_score(log_name, dataset_path, logs_dir, results_dir)
    evaluate_yield_prediction_score(log_name, dataset_path, logs_dir, results_dir)
    
    
eval_all_ChemLLMBench("mytest","/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/data/ChemLLMBench/chemllmbench","/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/data/ChemLLMBench_result","/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/eval/results")