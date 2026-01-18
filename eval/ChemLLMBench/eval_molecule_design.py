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

def evaluate_molecule_design_score(model_name, gt_path, logs_dir, results_dir):
    log_dir = f"{logs_dir}/molecule_design"
    
    if not os.path.exists(log_dir):
        raise ValueError(f"logs_dir {log_dir} is not correct")
    
    with open(f"{log_dir}/{model_name}.json", 'r') as f:
        samples = json.load(f)
    with open(f'{gt_path}/molecule_design/molecule_design.json') as f:
        gt_raw = json.load(f)
    
    preds = []
    gts = []
    for i, sample in enumerate(samples):
        meta = gt_raw[i]['meta']
        # Check if meta is already a dict or a JSON string
        if isinstance(meta, str):
            meta = json.loads(meta)
        gts.append(meta['reference'])

        # Handle both 'result' and 'results' keys depending on the data format
        if 'results' in sample:
            pred = extract_answer(sample['results'])
        elif 'result' in sample:
            pred = extract_answer(sample['result'])
        else:
            raise KeyError("Sample must contain either 'result' or 'results' key")

        preds.append(pred)
        
    res = evaluator.evaluate(preds, gts)
    fts = (res['rdk_sims'] + res['maccs_sims'] + res['morgan_sims']) / 3
    res['fts'] = fts
    
    os.makedirs(f"{results_dir}/molecule_design", exist_ok=True)
    json.dump(res, open(f"{results_dir}/molecule_design/eval_score_{model_name}.json", "w"), indent=4)
        
    return res