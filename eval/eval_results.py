from group_results import build_grouped_save_data
from argparse import ArgumentParser
import os
import json
import logging

logger = logging.getLogger()
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

parser = ArgumentParser()
parser.add_argument('--result_path', type=str, required=True)
parser.add_argument('--log_name', type=str, required=True)
parser.add_argument('--dataset_paths', type=str, nargs='+', required=True)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--mode', type=str, default='score')
args = parser.parse_args()

result_path = args.result_path
log_name = args.log_name
dataset_paths = args.dataset_paths
num_samples = args.num_samples
mode = args.mode

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
logs_dir = os.path.join(current_dir, 'logs')
results_dir = os.path.join(current_dir, 'results')
records_dir = os.path.join(current_dir, 'records')
print(results_dir)
os.makedirs(logs_dir, exist_ok=True)
build_grouped_save_data(result_path, logs_dir, log_name)

for dataset_path in dataset_paths:
    if 'ChemCoTBench' in dataset_path:
        from ChemCoTBench.eval_all import eval_all_ChemCoTBench, record_all_ChemCoTBench
        if mode == 'score':
            eval_all_ChemCoTBench(log_name, dataset_path, logs_dir, results_dir, num_samples)
        elif mode == 'record':
            record_all_ChemCoTBench(log_name, dataset_path, logs_dir, records_dir, num_samples)
    
    if 'ChemLLMBench' in dataset_path:
        from ChemLLMBench.eval_all import eval_all_ChemLLMBench, record_all_ChemLLMBench
        if mode == 'score':
            eval_all_ChemLLMBench(log_name, dataset_path, logs_dir, results_dir)
        elif mode == 'record':
            record_all_ChemLLMBench(log_name, dataset_path, logs_dir, records_dir)
    
    if 'ChemCoTDataset-test' in dataset_path:
        from ChemCoTDataset_textwise.eval_all import eval_all_ChemCoTDataset_textwise
        eval_all_ChemCoTDataset_textwise(log_name, dataset_path, logs_dir, results_dir, num_samples)
        
    if 'ChEBI' in dataset_path:
        from ChEBI20.eval_all import eval_all_ChEBI20, record_all_ChEBI20
        if mode == 'score':
            eval_all_ChEBI20(log_name, dataset_path, logs_dir, results_dir, num_samples)
        elif mode == 'record':
            record_all_ChEBI20(log_name, dataset_path, logs_dir, records_dir, num_samples)
        
    if 'InstructMol' in dataset_path:
        from InstructMol.eval_all import eval_all_InstructMol, record_all_InstructMol
        if mode == 'score':
            eval_all_InstructMol(log_name, dataset_path, logs_dir, results_dir, num_samples)
        elif mode == 'record':
            record_all_InstructMol(log_name, dataset_path, logs_dir, records_dir, num_samples)