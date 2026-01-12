import json
import os
import logging
from tqdm import tqdm
from ChemCoTBench.core.eval_metric import mol_opt_evaluater
from core.utils import extract_answer

def eval_molopt_from_list(optimized_prop, gt_list, pred_list, total_number):
    # this_function input: 
    #   gt_list for gt_molecules
    #   pred_list for pred_molecules
    #   total_number: len(gt_molecules)+len(cases that cannot extract SMILES)
    
    prop_dict = dict(logp='logp', solubility='solubility', qed="qed",  drd='drd2', jnk='jnk3', gsk='gsk3b')
    prop_evaluater = mol_opt_evaluater(prop=prop_dict[optimized_prop])
    
    improve_statistic = prop_evaluater.property_improvement(src_mol_list=gt_list, tgt_mol_list=pred_list, total_num=total_number)
    scaffold_hard, scaffold_soft = prop_evaluater.scaffold_consistency(src_mol_list=gt_list, tgt_mol_list=pred_list)
    
    result_dict = dict(
        improvement=improve_statistic,
        scaffold=dict(hard=scaffold_hard/total_number, soft=scaffold_soft/total_number),
    )
    
    return result_dict
    
def evaluate_molopt_score(model_name, gt_path, logs_dir, results_dir):
    ## 在get_molopt_cot中得到test结果, 我们评测这些test结果
    prop_dict = dict(logp='logp', solubility='solubility', qed="qed",  drd='drd2', jnk='jnk3', gsk='gsk3b')
    # prop_dict = dict(logp='logp', solubility='solubility', qed="qed",  drd='drd2', gsk='gsk3b')
    
    result_final = dict()
    
    logger = logging.getLogger(__name__)
    
    for prop in prop_dict.keys():
        logger.info(f'evaluating {prop} for model {model_name}')
        file_name = f"{logs_dir}/{prop}/{model_name}.json"
        pred_results = json.load(open(file_name, "r"))
        
                
        gt_name = f"{gt_path}/{prop}.json"
        gts = json.load(open(gt_name, "r"))
        
        tgt_smiles_list, src_smiles_list = list(), list()
        
        invalid_number = 0
        
        for i, pred in enumerate(pred_results):
            answer = extract_answer(pred['result'])
            if answer is None:
                invalid_number += 1
                continue
            tgt_smiles_list.append(answer)
            gt = gts[i]
            meta = json.loads(gt['meta'])
            src_smiles_list.append(meta['molecule'])
        
        logger.debug(len(pred_results), invalid_number, len(src_smiles_list))
        assert len(src_smiles_list) == len(tgt_smiles_list)
        assert len(pred_results) == invalid_number + len(src_smiles_list)
        
        result_dict = eval_molopt_from_list(optimized_prop=prop, gt_list=src_smiles_list, pred_list=tgt_smiles_list, total_number=len(pred_results))
        result_final[prop] = result_dict
    
    logger.info(f"eval_score_{model_name}_molopt:\n\r{result_final}")
    os.makedirs(f"{results_dir}/molopt", exist_ok=True)
    json.dump(result_final, open(f"{results_dir}/molopt/eval_score_{model_name}.json", "w"), indent=4)
    
    return result_final