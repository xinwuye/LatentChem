import os, json
from core.utils import extract_answer
import logging
import os
import sys
from core.evaluator import MoleculeCaptionEvaluator
evaluator = MoleculeCaptionEvaluator()
logger = logging.getLogger(__name__)

def evaluate_molecule_captioning_score(model_name, gt_path, logs_dir, results_dir):
    log_dir = f"{logs_dir}/molecule_captioning"
    
    if not os.path.exists(log_dir):
        raise ValueError(f"logs_dir {log_dir} is not correct")
    
    with open(f"{log_dir}/{model_name}.json", 'r') as f:
        samples = json.load(f)
    with open(f'{gt_path}/molecule_captioning/molecule_captioning.json') as f:
        gt_raw = json.load(f)
    
    preds = []
    gts = []
    for i, sample in enumerate(samples):
        gt = gt_raw[i]['gt']
        gts.append(gt)
        
        # Handle both 'result' and 'results' keys depending on the data format
        if 'results' in sample:
            pred = extract_answer(sample['results'])
        elif 'result' in sample:
            pred = extract_answer(sample['result'])
        else:
            raise KeyError("Sample must contain either 'result' or 'results' key")
        preds.append(pred)
        
    res = evaluator.evaluate(preds, gts)
    
    os.makedirs(f"{results_dir}/molecule_captioning", exist_ok=True)
    json.dump(res, open(f"{results_dir}/molecule_captioning/eval_score_{model_name}.json", "w"), indent=4)
        
    return res