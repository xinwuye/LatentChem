import sys, re, os, json
from ChemCoTBench.rxn.rxnutils import read_json, is_valid_smiles
from core.utils import extract_answer
import logging
import os

from core.evaluator import MoleculeSMILESEvaluator
evaluator = MoleculeSMILESEvaluator()
logger = logging.getLogger(__name__)

subtask_to_result_key = {
    "rcr": "SMILES",
    "nepp": "pred_smi",
    "mechsel": "choice",
    "major_product": "Major Product",
    "byproduct": "Byproduct(s)",
    "retro": "Reactants"
}

def evaluate_mol(model_name: str, subtask: str, gt_path, log_dir):
    log_dir = f"{log_dir}/{subtask}"
    
    if not os.path.exists(log_dir):
        raise ValueError(f"logs_dir {log_dir} is not correct")
    samples = read_json(f"{log_dir}/{model_name}.json")
    gt_raw = read_json(f'{gt_path}/{subtask}.json')
    preds = []
    gts = []
    for i, sample in enumerate(samples):
        gt = gt_raw[i]['gt']
        if subtask in ['major_product', 'byproduct']:
            gt = json.loads(gt)
            gts.append(gt.get(subtask_to_result_key[subtask], ''))
        # elif subtask == 'retro':
        #     if len(gt) == 0:
        #         continue
        #     gt = _combine_list(gt)
        #     gts.append(gt)
        else:
            gts.append(gt)

        try:
            pred_smiles = extract_answer(sample['result'])
            preds.append(pred_smiles)
        except Exception as e:
            logger.debug(f'error parsing {sample['result']}: {e}')
            preds.append('')
        
    res = evaluator.evaluate(preds, gts)
    fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
    res['fts'] = fts
        
    return res

def evaluate_MechSel(model_name: str, gt_path, logs_dir):
    """
    Evaluate the reaction mechanism selection prediction.

    Args:
        logs_dir (str): The directory where the logs are stored.
        model_name (str): The name of the model.

    Returns:
        None
    """
    logs_dir = f"{logs_dir}/mechsel"
    if not os.path.exists(logs_dir):
        raise ValueError(f"logs_dir {logs_dir} is not correct")
    samples = read_json(f"{logs_dir}/{model_name}.json")
    gt_raw = read_json(f'{gt_path}/mechsel.json')

    preds = []
    gts = []
    for i, sample in enumerate(samples):
        pred_choice = extract_answer(sample['result'])
        preds.append(pred_choice)
        # if pred_choice is not a valid choice, we treat it as empty
        if not pred_choice.lower().isalpha():
            pred_choice = ""

        pred_choice = pred_choice.lower()
        gt = gt_raw[i]['gt'].lower()
        preds.append(pred_choice)
        gts.append(gt)

    accuracy = sum(1 for pred, gt in zip(preds, gts) if pred == gt) / len(gts)
    return {"MCQ Accuracy (mean)": accuracy}

def evaluate_rxn_score(model_name: str, gt_path: str , logs_dir: str, results_dir):
    all_results = {}
    subtasks = subtask_to_result_key.keys()
    for subtask in subtasks:
        logger.info(f'evaluating {subtask} for model {model_name}')
        try:
            if subtask == 'MechSel' or subtask == 'mechsel':
                all_results[subtask] = evaluate_MechSel(model_name, gt_path, logs_dir)
            else:
                all_results[subtask] = evaluate_mol(model_name, subtask, gt_path, logs_dir)
        except Exception as e:
            logger.error(f"Error evaluating {subtask} for {model_name}: {e}")
            all_results[subtask] = None
    logger.info(f"eval_score_{model_name}_rxn:\n\r{all_results}")
    os.makedirs(f"{results_dir}/rxn", exist_ok=True)
    json.dump(all_results, open(f"{results_dir}/rxn/eval_score_{model_name}.json", "w"), indent=4)

    return all_results
