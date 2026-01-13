"""
Unified runner script for InstructMol dataset evaluation.

This script runs the complete evaluation pipeline:
1. Preprocessing (if needed)
2. Inference
3. Evaluation

Supports both ChEBI-20 and Molecule-Oriented Instructions datasets.
"""

import os
import sys
import argparse
import subprocess
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    if description:
        logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False


def preprocess_chebi20(args):
    """Run ChEBI-20 preprocessing."""
    logger.info("\n" + "="*80)
    logger.info("Step 1: Preprocessing ChEBI-20 dataset")
    logger.info("="*80)

    if args.skip_preprocessing and os.path.exists(args.chebi20_processed_path):
        logger.info(f"Skipping preprocessing (file exists): {args.chebi20_processed_path}")
        return True

    cmd = [
        "python3",
        "eval/InstructMol/preprocess_chebi20.py",
        "--csv_path", args.chebi20_csv_path,
        "--output_dir", os.path.dirname(args.chebi20_processed_path),
        "--split", "test"
    ]

    return run_command(cmd, "ChEBI-20 preprocessing")


def preprocess_molecule_instructions(args):
    """Run Molecule-Oriented Instructions preprocessing."""
    logger.info("\n" + "="*80)
    logger.info("Step 1: Preprocessing Molecule-Oriented Instructions dataset")
    logger.info("="*80)

    if args.skip_preprocessing and os.path.exists(args.mol_instr_processed_dir):
        logger.info(f"Skipping preprocessing (directory exists): {args.mol_instr_processed_dir}")
        return True

    cmd = [
        "python3",
        "eval/InstructMol/preprocess_molecule_instructions.py",
        "--input_dir", args.mol_instr_raw_dir,
        "--output_dir", args.mol_instr_processed_dir,
    ]

    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])

    return run_command(cmd, "Molecule-Oriented Instructions preprocessing")


def run_inference(args, dataset_path, output_path):
    """Run Bio-LatentCOT inference."""
    logger.info("\n" + "="*80)
    logger.info(f"Step 2: Running inference on {dataset_path}")
    logger.info("="*80)

    if args.skip_inference and os.path.exists(output_path):
        logger.info(f"Skipping inference (file exists): {output_path}")
        return True

    cmd = [
        "python3",
        "code_train_sft/inference.py",
        "--data_path", dataset_path,
        "--lora_path", args.lora_path,
        "--inference_results_path", output_path,
        "--batch_size", str(args.batch_size),
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
    ]

    # Add projector_path only if provided
    if args.projector_path is not None:
        cmd.extend(["--projector_path", args.projector_path])

    if args.max_test_samples:
        cmd.extend(["--max_test_samples", str(args.max_test_samples)])

    return run_command(cmd, f"Inference on {dataset_path}")


def run_evaluation_chebi20(args):
    """Run ChEBI-20 evaluation."""
    logger.info("\n" + "="*80)
    logger.info("Step 3: Evaluating ChEBI-20 results")
    logger.info("="*80)

    cmd = [
        "python3",
        "eval/InstructMol/eval_chebi20.py",
        "--results_path", args.chebi20_results_path,
        "--gt_path", args.chebi20_processed_path,
        "--output_dir", args.output_dir
    ]

    return run_command(cmd, "ChEBI-20 evaluation")


def run_evaluation_molecule_instructions(args):
    """Run Molecule-Oriented Instructions evaluation."""
    logger.info("\n" + "="*80)
    logger.info("Step 3: Evaluating Molecule-Oriented Instructions results")
    logger.info("="*80)

    cmd = [
        "python3",
        "eval/InstructMol/eval_molecule_instructions.py",
        "--mode", "all",
        "--results_dir", args.mol_instr_results_dir,
        "--gt_dir", args.mol_instr_processed_dir,
        "--output_dir", args.output_dir
    ]

    return run_command(cmd, "Molecule-Oriented Instructions evaluation")


def run_chebi20_pipeline(args):
    """Run complete ChEBI-20 evaluation pipeline."""
    logger.info("\n" + "#"*80)
    logger.info("Starting ChEBI-20 Evaluation Pipeline")
    logger.info("#"*80)

    # Step 1: Preprocessing
    if not preprocess_chebi20(args):
        logger.error("ChEBI-20 preprocessing failed!")
        return False

    # Step 2: Inference
    if not run_inference(args, args.chebi20_processed_path, args.chebi20_results_path):
        logger.error("ChEBI-20 inference failed!")
        return False

    # Step 3: Evaluation
    if not run_evaluation_chebi20(args):
        logger.error("ChEBI-20 evaluation failed!")
        return False

    logger.info("\n" + "#"*80)
    logger.info("ChEBI-20 Evaluation Pipeline Complete!")
    logger.info("#"*80)
    return True


def run_molecule_instructions_pipeline(args):
    """Run complete Molecule-Oriented Instructions evaluation pipeline."""
    logger.info("\n" + "#"*80)
    logger.info("Starting Molecule-Oriented Instructions Evaluation Pipeline")
    logger.info("#"*80)

    # Step 1: Preprocessing
    if not preprocess_molecule_instructions(args):
        logger.error("Molecule-Oriented Instructions preprocessing failed!")
        return False

    # Step 2: Inference for each task
    tasks = [
        'molecular_description_generation',
        'forward_reaction_prediction',
        'retrosynthesis',
        'reagent_prediction'
    ]

    os.makedirs(args.mol_instr_results_dir, exist_ok=True)

    for task in tasks:
        task_data_path = os.path.join(args.mol_instr_processed_dir, f"{task}.json")
        if not os.path.exists(task_data_path):
            logger.warning(f"Task data not found: {task_data_path}, skipping...")
            continue

        task_results_path = os.path.join(args.mol_instr_results_dir, f"{task}_inference.json")

        if not run_inference(args, task_data_path, task_results_path):
            logger.error(f"Inference failed for task: {task}")
            # Continue with other tasks instead of failing completely
            continue

    # Step 3: Evaluation
    if not run_evaluation_molecule_instructions(args):
        logger.error("Molecule-Oriented Instructions evaluation failed!")
        return False

    logger.info("\n" + "#"*80)
    logger.info("Molecule-Oriented Instructions Evaluation Pipeline Complete!")
    logger.info("#"*80)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run InstructMol dataset evaluation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["chebi20", "molecule_instructions", "both"],
        required=True,
        help="Which dataset(s) to evaluate"
    )

    # Model paths
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/mnt/afs/L202500070/Bio-LatentCOT/code_train_sft/outputs/stage3-lr2e-4-cf_margin01/stage3/lora_weights",
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--projector_path",
        type=str,
        default=None,
        help="Path to projector weights (optional)"
    )

    # ChEBI-20 paths
    parser.add_argument(
        "--chebi20_csv_path",
        type=str,
        default="instructmol/ChEBI-20/test_with_smiles.csv",
        help="Path to ChEBI-20 CSV file"
    )
    parser.add_argument(
        "--chebi20_processed_path",
        type=str,
        default="instructmol/ChEBI-20/processed/chebi20_test.json",
        help="Path to processed ChEBI-20 JSON"
    )
    parser.add_argument(
        "--chebi20_results_path",
        type=str,
        default="outputs/instructmol/chebi20_inference.json",
        help="Path to save ChEBI-20 inference results"
    )

    # Molecule-Oriented Instructions paths
    parser.add_argument(
        "--mol_instr_raw_dir",
        type=str,
        default="instructmol/Molecule-oriented_Instructions",
        help="Directory with raw Molecule-Oriented Instructions JSON files"
    )
    parser.add_argument(
        "--mol_instr_processed_dir",
        type=str,
        default="instructmol/Molecule-oriented_Instructions/processed",
        help="Directory for processed Molecule-Oriented Instructions JSON files"
    )
    parser.add_argument(
        "--mol_instr_results_dir",
        type=str,
        default="outputs/instructmol/molecule_instructions",
        help="Directory to save Molecule-Oriented Instructions inference results"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/instructmol",
        help="Directory for evaluation results"
    )

    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples (for testing)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples for preprocessing (for testing)")

    # Pipeline control
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing if files exist")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference if files exist")

    args = parser.parse_args()

    # Run the requested pipeline(s)
    success = True

    if args.dataset in ["chebi20", "both"]:
        if not run_chebi20_pipeline(args):
            success = False

    if args.dataset in ["molecule_instructions", "both"]:
        if not run_molecule_instructions_pipeline(args):
            success = False

    if success:
        logger.info("\n" + "🎉"*40)
        logger.info("All evaluation pipelines completed successfully!")
        logger.info("🎉"*40)
    else:
        logger.error("\n" + "❌"*40)
        logger.error("Some evaluation pipelines failed. Check the logs above.")
        logger.error("❌"*40)
        sys.exit(1)


if __name__ == "__main__":
    main()
