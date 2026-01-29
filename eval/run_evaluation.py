from group_results import build_grouped_save_data
import os
import logging

logger = logging.getLogger()
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Patch the evaluation functions to look in the correct subdirectories
original_eval_functions = {}

def patch_evaluation_functions():
    """Patch the evaluation functions to look for ground truth files in subdirectories."""
    import ChemLLMBench.eval_molecule_captioning
    import ChemLLMBench.eval_molecule_design
    import ChemLLMBench.eval_name_prediction
    import ChemLLMBench.eval_property_prediction
    import ChemLLMBench.eval_reaction_prediction
    import ChemLLMBench.eval_reagent_selection
    import ChemLLMBench.eval_retro
    import ChemLLMBench.eval_yield_prediction

    # Store original functions
    original_eval_functions['molecule_captioning'] = ChemLLMBench.eval_molecule_captioning.evaluate_molecule_captioning_score
    original_eval_functions['molecule_design'] = ChemLLMBench.eval_molecule_design.evaluate_molecule_design_score
    original_eval_functions['name_prediction'] = ChemLLMBench.eval_name_prediction.evaluate_name_prediction_score
    original_eval_functions['property_prediction'] = ChemLLMBench.eval_property_prediction.evaluate_property_prediction_score
    original_eval_functions['reaction_prediction'] = ChemLLMBench.eval_reaction_prediction.evaluate_reaction_prediction_score
    original_eval_functions['reagent_selection'] = ChemLLMBench.eval_reagent_selection.evaluate_reagent_selection_score
    original_eval_functions['retro'] = ChemLLMBench.eval_retro.evaluate_retro_score
    original_eval_functions['yield_prediction'] = ChemLLMBench.eval_yield_prediction.evaluate_yield_prediction_score

    # Define patched functions
    def patched_evaluate_molecule_captioning_score(model_name, gt_path, logs_dir, results_dir):
        # Update gt_path to point to the subdirectory
        gt_subdir = os.path.join(gt_path, 'molecule_captioning')
        return original_eval_functions['molecule_captioning'](model_name, gt_subdir, logs_dir, results_dir)

    def patched_evaluate_molecule_design_score(model_name, gt_path, logs_dir, results_dir):
        gt_subdir = os.path.join(gt_path, 'molecule_design')
        return original_eval_functions['molecule_design'](model_name, gt_subdir, logs_dir, results_dir)

    def patched_evaluate_name_prediction_score(model_name, gt_path, logs_dir, results_dir):
        gt_subdir = os.path.join(gt_path, 'name_prediction')  # assuming this subdirectory exists
        return original_eval_functions['name_prediction'](model_name, gt_subdir, logs_dir, results_dir)

    def patched_evaluate_property_prediction_score(model_name, gt_path, logs_dir, results_dir):
        gt_subdir = os.path.join(gt_path, 'property_prediction')  # assuming this subdirectory exists
        return original_eval_functions['property_prediction'](model_name, gt_subdir, logs_dir, results_dir)

    def patched_evaluate_reaction_prediction_score(model_name, gt_path, logs_dir, results_dir):
        gt_subdir = os.path.join(gt_path, 'reaction_prediction')
        return original_eval_functions['reaction_prediction'](model_name, gt_subdir, logs_dir, results_dir)

    def patched_evaluate_reagent_selection_score(model_name, gt_path, logs_dir, results_dir):
        # This might have multiple subtasks
        gt_subdir = os.path.join(gt_path, 'reagent_selection')
        return original_eval_functions['reagent_selection'](model_name, gt_subdir, logs_dir, results_dir)

    def patched_evaluate_retro_score(model_name, gt_path, logs_dir, results_dir):
        gt_subdir = os.path.join(gt_path, 'retro')
        return original_eval_functions['retro'](model_name, gt_subdir, logs_dir, results_dir)

    def patched_evaluate_yield_prediction_score(model_name, gt_path, logs_dir, results_dir):
        gt_subdir = os.path.join(gt_path, 'yield_prediction')  # assuming this subdirectory exists
        return original_eval_functions['yield_prediction'](model_name, gt_subdir, logs_dir, results_dir)

    # Apply patches
    ChemLLMBench.eval_molecule_captioning.evaluate_molecule_captioning_score = patched_evaluate_molecule_captioning_score
    ChemLLMBench.eval_molecule_design.evaluate_molecule_design_score = patched_evaluate_molecule_design_score
    ChemLLMBench.eval_name_prediction.evaluate_name_prediction_score = patched_evaluate_name_prediction_score
    ChemLLMBench.eval_property_prediction.evaluate_property_prediction_score = patched_evaluate_property_prediction_score
    ChemLLMBench.eval_reaction_prediction.evaluate_reaction_prediction_score = patched_evaluate_reaction_prediction_score
    ChemLLMBench.eval_reagent_selection.evaluate_reagent_selection_score = patched_evaluate_reagent_selection_score
    ChemLLMBench.eval_retro.evaluate_retro_score = patched_evaluate_retro_score
    ChemLLMBench.eval_yield_prediction.evaluate_yield_prediction_score = patched_evaluate_yield_prediction_score

# Hardcoded paths for your specific evaluation
result_path = "/BioLatent/Bio-LatentCOT/eval/Mol-Instructions/stage2_with_cot/mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage2_with_cot/inference/results_20260110-050429_p0.json"
log_name = "results_20260110-050429_p0"
dataset_path = "/BioLatent/Bio-LatentCOT/data/ChemLLMBench/chemllmbench"

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
logs_dir = os.path.join(current_dir, 'logs')
results_dir = os.path.join(current_dir, 'results')

os.makedirs(logs_dir, exist_ok=True)
build_grouped_save_data(result_path, logs_dir, log_name)

# Apply patches before running evaluation
patch_evaluation_functions()

if 'ChemLLMBench' in dataset_path:
    from ChemLLMBench.eval_all import eval_all_ChemLLMBench
    eval_all_ChemLLMBench(log_name, dataset_path, logs_dir, results_dir)