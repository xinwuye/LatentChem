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
args = parser.parse_args()

result_path = args.result_path
log_name = args.log_name
dataset_paths = args.dataset_paths

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
logs_dir = os.path.join(current_dir, 'logs')
results_dir = os.path.join(current_dir, 'results')

os.makedirs(logs_dir, exist_ok=True)
build_grouped_save_data(result_path, logs_dir, log_name)

for dataset_path in dataset_paths:
    if 'ChemCoTBench' in dataset_path:
        from ChemCoTBench.eval_all import eval_all_ChemCoTBench
        eval_all_ChemCoTBench(log_name, dataset_path, logs_dir, results_dir)
    
    if 'ChemLLMBench' in dataset_path:
        from ChemLLMBench.eval_all import eval_all_ChemLLMBench
        eval_all_ChemLLMBench(log_name, dataset_path, logs_dir, results_dir)