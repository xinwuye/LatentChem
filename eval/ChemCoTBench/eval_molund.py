
import json
import os
import logging
from ChemCoTBench.core.eval_metric import mol_opt_evaluater
from core.utils import extract_answer

def eval_molund_from_list(gt_list, pred_list, total_number, task):
    # this_function input: 
    #   gt_list for gt_molecules
    #   pred_list for pred_molecules
    #   total_number: len(gt_molecules)+len(cases that cannot extract SMILES)
    score = None
    if task in ["ring_system", "ring_system_scaffold", "permutated"]: # "ring_system_scaffold", "ring_system" as the same
        count = sum(1 for item in pred_list if str(item).lower() == "yes")
        if len(pred_list) == 0: score = None
        else: score = count / len(pred_list)
    elif task == "mutated":
        count = sum(1 for item in pred_list if str(item).lower() == "no")
        if len(pred_list) == 0: score = None
        else: score = count / len(pred_list)
    elif task == 'equivalence': # "equivalence" = "mutated" + "permutated"
        count = 0
        for i in range(len(pred_list)):
            if str(pred_list[i]).lower() == str(gt_list[i]).lower():
                count += 1
        if len(pred_list) == 0: score = None
        else: score = count / len(pred_list)
    elif task in ["ring_count", "fg_samples", "fg_count"]:
        assert len(gt_list) == len(pred_list)
        if len(gt_list) == 0: score = None
        else: score = sum([abs(int(pred_list[i])-int(gt_list[i])) for i in range(len(pred_list))]) / len(gt_list)
    elif task in ["murcko", "Murcko_scaffold"]: # "murcko" "Murcko_scaffold" are the same
        assert len(gt_list) == len(pred_list)
        prop_evaluater = mol_opt_evaluater(prop='qed')
        scaffold_hard, scaffold_soft = prop_evaluater.scaffold_consistency(src_mol_list=gt_list, tgt_mol_list=pred_list)
        if len(gt_list) == 0: score = None
        else: score = scaffold_soft / len(pred_list)
    
    my_dict = {
        "score": score,
        f"{task}-valid-rate": len(pred_list)/total_number
    }
    
    return my_dict
    
def evaluate_molund_score(model_name, gt_path, logs_dir, results_dir):
    logger = logging.getLogger(__name__)
    task_dict = dict(
        fg_samples="fg_count", murcko='Murcko_scaffold', ring_count='ring_count',
        ring_system='ring_system_scaffold', equivalence = 'equivalence'
    )
    
    result_dict = dict()
    
    for task in task_dict.keys():
        logger.info(f'evaluating {task} for model {model_name}')
        if 'llama' not in model_name:
            file_name = f"{logs_dir}/{task_dict[task]}/{model_name}.json"
            
        pred_results = json.load(open(file_name, "r"))
        invalid_number = 0
        
        gt_name = f"{gt_path}/{task_dict[task]}.json"
        gts = json.load(open(gt_name, "r"))
        
        pred_list, gt_list = list(), list()
        for i, pred in enumerate(pred_results):
            answer = extract_answer(pred['result'])
            if answer is None:
                invalid_number += 1
                continue
            pred_list.append(answer)
            gt_list.append(gts[i]['gt'])
        
        assert len(pred_results) == invalid_number+len(pred_list)
        result_dict[task] = eval_molund_from_list(gt_list=gt_list, pred_list=pred_list, total_number=len(pred_results), task=task)
        logger.debug(model_name, task, result_dict[task])
    
    logger.info(f"eval_score_{model_name}_molund:\n\r{result_dict}")
    os.makedirs(f"{results_dir}/molund", exist_ok=True)
    json.dump(result_dict, open(f"{results_dir}/molund/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_dict        