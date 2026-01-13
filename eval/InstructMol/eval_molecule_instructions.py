"""
Evaluation script for Molecule-Oriented Instructions dataset.

This script evaluates model outputs on multiple tasks from the
Molecule-Oriented Instructions dataset.
"""

import json
import os
import logging
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import core utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.evaluator import MoleculeCaptionEvaluator, MoleculeSMILESEvaluator
from core.utils import extract_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TASK_CONFIGS = {
    'molecular_description_generation': {
        'evaluator': 'caption',
        'metrics': ["bleu-2", "bleu-4", "meteor", "rouge-1", "rouge-2", "rouge-L"]
    },
    'forward_reaction_prediction': {
        'evaluator': 'smiles',
        'metrics': ["exact_match", "bleu", "levenshtein", "validity", "maccs_sims", "morgan_sims", "rdk_sims"]
    },
    'retrosynthesis': {
        'evaluator': 'smiles',
        'metrics': ["exact_match", "bleu", "levenshtein", "validity", "maccs_sims", "morgan_sims", "rdk_sims"]
    },
    'reagent_prediction': {
        'evaluator': 'smiles',
        'metrics': ["exact_match", "bleu", "levenshtein", "validity", "maccs_sims", "morgan_sims", "rdk_sims"]
    }
}


def load_inference_results(results_path):
    """Load inference results from JSON file."""
    logger.info(f"Loading inference results from {results_path}")

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_results = data.get('test_results', [])
    logger.info(f"Loaded {len(test_results)} inference results")

    return test_results


def load_ground_truth(gt_path):
    """Load ground truth data from processed JSON."""
    logger.info(f"Loading ground truth from {gt_path}")

    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create a lookup dictionary by sample index
    gt_dict = {}
    for idx, item in enumerate(data):
        item_id = item.get('id', f'sample_{idx}')
        meta = json.loads(item['meta'])
        gt_value = meta.get('gt', '')
        gt_dict[str(idx)] = {
            'id': item_id,
            'gt': gt_value,
            'meta': meta
        }

    logger.info(f"Loaded {len(gt_dict)} ground truth samples")
    return gt_dict


def evaluate_task(results_path, gt_path, task_name, output_dir):
    """
    Evaluate a specific task from Molecule-Oriented Instructions.

    Args:
        results_path: Path to inference results JSON
        gt_path: Path to ground truth JSON
        task_name: Name of the task
        output_dir: Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    if task_name not in TASK_CONFIGS:
        logger.error(f"Unknown task: {task_name}")
        return None

    config = TASK_CONFIGS[task_name]

    # Load data
    inference_results = load_inference_results(results_path)
    gt_dict = load_ground_truth(gt_path)

    # Extract predictions and ground truths
    predictions = []
    references = []
    valid_count = 0
    invalid_count = 0

    for result in inference_results:
        # Extract prediction from model output
        raw_output = result.get('result', '')
        pred_text = extract_answer(raw_output)

        if pred_text is None:
            # If no <answer> tags, use the raw output
            pred_text = raw_output.strip()

        # Get ground truth by sample index
        sample_id = str(result.get('sample_id', ''))

        if sample_id not in gt_dict:
            invalid_count += 1
            continue

        gt_info = gt_dict[sample_id]
        gt_value = gt_info['gt']

        predictions.append(pred_text)
        references.append(gt_value)
        valid_count += 1

    logger.info(f"Task: {task_name}")
    logger.info(f"  Valid predictions: {valid_count}")
    logger.info(f"  Invalid/missing: {invalid_count}")

    if valid_count == 0:
        logger.error(f"No valid predictions to evaluate for {task_name}!")
        return None

    # Select evaluator based on task type
    if config['evaluator'] == 'caption':
        evaluator = MoleculeCaptionEvaluator()
    else:  # 'smiles'
        evaluator = MoleculeSMILESEvaluator()

    # Evaluate
    logger.info(f"Computing evaluation metrics for {task_name}...")
    try:
        results = evaluator.evaluate(
            predictions,
            references,
            metrics=config['metrics'],
            verbose=False
        )
    except Exception as e:
        logger.error(f"Evaluation failed for {task_name}: {e}")
        return None

    # Log results
    logger.info("\n" + "="*80)
    logger.info(f"Evaluation Results: {task_name}")
    logger.info("="*80)
    for metric, value in results.items():
        logger.info(f"{metric:15s}: {value:.4f}")
    logger.info("="*80)

    # Save results
    output_file = os.path.join(output_dir, f"eval_{task_name}_results.json")
    results_with_meta = {
        "dataset": "Molecule-Oriented Instructions",
        "task": task_name,
        "num_samples": valid_count,
        "metrics": results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_with_meta, f, indent=2)

    logger.info(f"\nResults saved to {output_file}\n")

    return results


def evaluate_all_tasks(results_dir, gt_dir, output_dir):
    """
    Evaluate all tasks from Molecule-Oriented Instructions.

    Args:
        results_dir: Directory containing inference results for each task
        gt_dir: Directory containing ground truth JSON files for each task
        output_dir: Directory to save evaluation results
    """
    all_results = {}

    for task_name in TASK_CONFIGS.keys():
        results_path = os.path.join(results_dir, f"{task_name}_inference.json")
        gt_path = os.path.join(gt_dir, f"{task_name}.json")

        if not os.path.exists(results_path):
            logger.warning(f"Results file not found: {results_path}, skipping...")
            continue

        if not os.path.exists(gt_path):
            logger.warning(f"Ground truth file not found: {gt_path}, skipping...")
            continue

        task_results = evaluate_task(results_path, gt_path, task_name, output_dir)
        if task_results:
            all_results[task_name] = task_results

    # Save summary
    if all_results:
        summary_file = os.path.join(output_dir, "eval_all_tasks_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

        logger.info("\n" + "="*80)
        logger.info("Summary of All Tasks")
        logger.info("="*80)
        for task, metrics in all_results.items():
            logger.info(f"\n{task}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric:15s}: {value:.4f}")
        logger.info("="*80)
        logger.info(f"\nSummary saved to {summary_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Molecule-Oriented Instructions tasks"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "all"],
        default="single",
        help="Evaluation mode: single task or all tasks"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="Path to inference results JSON file (for single mode)"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        help="Path to ground truth JSON file (for single mode)"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        choices=list(TASK_CONFIGS.keys()),
        help="Task name (for single mode)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory with inference results (for all mode)"
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        help="Directory with ground truth files (for all mode)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results/molecule_instructions",
        help="Output directory for evaluation results"
    )

    args = parser.parse_args()

    if args.mode == "single":
        if not all([args.results_path, args.gt_path, args.task_name]):
            parser.error("For single mode, --results_path, --gt_path, and --task_name are required")

        evaluate_task(
            args.results_path,
            args.gt_path,
            args.task_name,
            args.output_dir
        )
    else:  # all
        if not all([args.results_dir, args.gt_dir]):
            parser.error("For all mode, --results_dir and --gt_dir are required")

        evaluate_all_tasks(
            args.results_dir,
            args.gt_dir,
            args.output_dir
        )


if __name__ == "__main__":
    main()
