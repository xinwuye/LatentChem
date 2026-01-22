import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse

def is_desc(task_name, metric_name):
    desc = False
    # ChemCoTBench
    desc = desc or task_name in ['ring_count', 'fg_count'] and metric_name == 'score'
    desc = desc or metric_name == 'levenshtein'
    return desc

# ChemCoTBench
# metric_of_interest = ['correct_rate',
#                       'mean', 'success_rate',
#                       'score',
#                       'exact_match', 'fts']

# InstructMol
metric_of_interest = ['exact_match',
                      'bleu', 'levenshtein',
                      'rdk_sims', 'maccs_sims', 'morgan_sims',
                      'validity',
                      'bleu-2', 'bleu-4', 'meteor',
                      'rouge-1', 'rouge-2', 'rouge-L']

def eval_single_pair(task_name, dir, model_A, model_B):
    try:
        df_A = pd.read_csv(os.path.join(dir, 'eval_results_' + model_A + '.csv'))
        # print(os.path.join(dir, 'eval_results_' + model_A + '.csv'))
        df_B = pd.read_csv(os.path.join(dir, 'eval_results_' + model_B + '.csv'))
    except:
        return 0, 0

    win_count = 0
    tie_count = 0
    loss_count = 0
    count = 0
    for col in df_A.columns:
        if col in metric_of_interest:
            desc = is_desc(task_name, col)
            series_A = df_A[col]
            series_B = df_B[col]
            win_count += (series_A > series_B if not desc else series_A < series_B).sum()
            tie_count += (series_A == series_B).sum()
            loss_count += (series_A > series_B if desc else series_A < series_B).sum()
            count += len(series_A)
    
    non_tie_count = win_count + loss_count
    if non_tie_count == 0:
        print(task_name, 'No non-tie cases')
        return 0, 0
    print(task_name ,win_count * 1.0 / non_tie_count, loss_count * 1.0 / non_tie_count)
    return win_count, loss_count
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--model_A', type=str, required=True)
    parser.add_argument('--model_B', type=str, required=True)
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    all_task_names = []
    all_task_dirs = []
    for csv_file in root_dir.rglob('*.csv'):
        subtask_path = csv_file.parent
        subtask_name = subtask_path.name
        
        if not subtask_name in all_task_names:
            all_task_names.append(subtask_name)
            all_task_dirs.append(subtask_path)
    
    total_win_count = 0
    total_loss_count = 0
    for task, dir in zip(all_task_names, all_task_dirs):
        win_count, loss_count = eval_single_pair(task, dir, args.model_A, args.model_B)
        
        total_win_count += win_count
        total_loss_count += loss_count
        
    print(total_win_count, total_loss_count)