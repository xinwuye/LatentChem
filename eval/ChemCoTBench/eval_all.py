from ChemCoTBench.eval_moledit import evaluate_moledit_score
from ChemCoTBench.eval_molopt import evaluate_molopt_score
from ChemCoTBench.eval_molund import evaluate_molund_score
from ChemCoTBench.eval_rxn import evaluate_rxn_score

def eval_all_ChemCoTBench(log_name, dataset_path, logs_dir, results_dir):
    evaluate_moledit_score(log_name, f'{dataset_path}/mol_edit', logs_dir, results_dir)
    evaluate_molopt_score(log_name, f'{dataset_path}/mol_opt', logs_dir, results_dir)
    evaluate_molund_score(log_name, f'{dataset_path}/mol_und', logs_dir, results_dir)
    evaluate_rxn_score(log_name, f'{dataset_path}/rxn', logs_dir, results_dir)