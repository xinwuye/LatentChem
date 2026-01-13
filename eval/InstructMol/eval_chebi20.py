"""
Evaluation script for ChEBI-20 dataset (Molecule Description Generation).

This script evaluates model outputs on the ChEBI-20 dataset using metrics
aligned with the InstructMol paper (BLEU, METEOR, ROUGE).
"""

import json
import os
import logging
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import core utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.evaluator import MoleculeCaptionEvaluator
from core.utils import extract_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_inference_results(results_path):
    """
    Load inference results from JSON file.

    Expected format:
    {
        "test_results": [
            {
                "sample_id": ...,
                "smiles": [...],
                "task": "...",
                "result": "model generated text"
            },
            ...
        ]
    }
    """
    logger.info(f"Loading inference results from {results_path}")

    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_results = data.get('test_results', [])
    logger.info(f"Loaded {len(test_results)} inference results")

    return test_results


def load_ground_truth(gt_path):
    """
    Load ground truth data from processed ChEBI-20 JSON.

    Expected format:
    [
        {
            "id": "chebi20_test_...",
            "query": "...",
            "meta": "{\"molecule\": \"...\", \"gt\": \"...\" ...}",
            ...
        },
        ...
    ]
    """
    logger.info(f"Loading ground truth from {gt_path}")

    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract ground truth descriptions
    gt_dict = {}
    for item in data:
        item_id = item['id']
        meta = json.loads(item['meta'])
        gt_text = meta.get('gt', '')
        gt_dict[item_id] = gt_text

    logger.info(f"Loaded {len(gt_dict)} ground truth samples")
    return gt_dict


def evaluate_chebi20(results_path, gt_path, output_dir):
    """
    Evaluate ChEBI-20 model outputs.

    Args:
        results_path: Path to inference results JSON
        gt_path: Path to ground truth JSON (processed ChEBI-20)
        output_dir: Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    inference_results = load_inference_results(results_path)
    gt_dict = load_ground_truth(gt_path)

    # Extract predictions and ground truths
    predictions = []
    references = []
    valid_count = 0
    missing_gt_count = 0

    for result in inference_results:
        # Extract prediction from model output
        raw_output = result.get('result', '')
        pred_text = extract_answer(raw_output)

        if pred_text is None:
            # If no <answer> tags, use the raw output
            pred_text = raw_output.strip()

        # Get ground truth
        sample_id = result.get('sample_id', '')

        # Try to match with ground truth
        # The sample_id might be an integer, so we need to find the corresponding entry
        gt_text = None

        # Try direct lookup
        if str(sample_id) in gt_dict:
            gt_text = gt_dict[str(sample_id)]
        else:
            # Try to find by index if sample_id is numeric
            try:
                idx = int(sample_id)
                # Look for entries with this index in their ID
                for key in gt_dict:
                    if f"_{idx}" in key or key.endswith(str(idx)):
                        gt_text = gt_dict[key]
                        break
            except (ValueError, KeyError):
                pass

        if gt_text is None:
            missing_gt_count += 1
            continue

        predictions.append(pred_text)
        references.append(gt_text)
        valid_count += 1

    logger.info(f"Valid predictions: {valid_count}")
    logger.info(f"Missing ground truth: {missing_gt_count}")

    if valid_count == 0:
        logger.error("No valid predictions to evaluate!")
        return None

    # Evaluate using MoleculeCaptionEvaluator
    evaluator = MoleculeCaptionEvaluator()
    metrics = ["bleu-2", "bleu-4", "meteor", "rouge-1", "rouge-2", "rouge-L"]

    logger.info("Computing evaluation metrics...")
    results = evaluator.evaluate(predictions, references, metrics=metrics)

    # Log results
    logger.info("\n" + "="*80)
    logger.info("ChEBI-20 Evaluation Results (Molecule Description Generation)")
    logger.info("="*80)
    for metric, value in results.items():
        logger.info(f"{metric:15s}: {value:.4f}")
    logger.info("="*80)

    # Save results
    output_file = os.path.join(output_dir, "eval_chebi20_results.json")
    results_with_meta = {
        "dataset": "ChEBI-20",
        "task": "Molecule Description Generation",
        "num_samples": valid_count,
        "metrics": results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_with_meta, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChEBI-20 (Molecule Description Generation)")
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to inference results JSON file"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        required=True,
        help="Path to processed ground truth JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results/chebi20",
        help="Output directory for evaluation results"
    )

    args = parser.parse_args()

    evaluate_chebi20(args.results_path, args.gt_path, args.output_dir)


if __name__ == "__main__":
    main()
