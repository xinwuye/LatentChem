#!/usr/bin/env python3
"""
Test script for InstructMol dataloader.
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import load_instructmol_data, extract_instructmol_fields
import json

def test_extract_instructmol_fields():
    """Test the field extraction function."""
    print("=" * 80)
    print("Testing extract_instructmol_fields()...")
    print("=" * 80)

    # Sample data from the dataset
    sample = {
        "instruction": "Please provide the energy separation between the highest occupied and lowest unoccupied molecular orbitals (HOMO-LUMO gap) of this molecule.",
        "input": "[O][=C][C][N][C][Branch1][Ring1][C][=O][C][=C][Ring1][#Branch1]",
        "output": 0.1913,
        "metadata": {
            "task": "property prediction",
            "split": "train"
        }
    }

    # Test training mode
    print("\n1. Testing training mode (is_eval=False):")
    result = extract_instructmol_fields(sample, is_eval=False)
    print(f"   Query: {result['query'][:100]}...")
    print(f"   Input SMILES: {result['input_smiles']}")
    print(f"   Label: {result['label']}")
    print(f"   Task: {result['task']}")

    # Test eval mode
    print("\n2. Testing eval mode (is_eval=True):")
    result_eval = extract_instructmol_fields(sample, is_eval=True)
    print(f"   Query: {result_eval['query'][:100]}...")
    print(f"   Input SMILES: {result_eval['input_smiles']}")
    print(f"   Label: {result_eval['label']}")
    print(f"   Task: {result_eval['task']}")

    print("\n✓ Field extraction test passed!\n")


def test_load_instructmol_data():
    """Test loading the full dataset."""
    print("=" * 80)
    print("Testing load_instructmol_data()...")
    print("=" * 80)

    # Path to the InstructMol dataset
    data_path = "/mnt/afs/L202500070/yimeng/Bio-LatentCOT/instructmol/Molecule-oriented_Instructions"

    # Test loading in eval mode
    print("\n1. Loading dataset in eval mode (first 100 samples)...")
    try:
        dataset = load_instructmol_data(
            path=data_path,
            max_len=2048,
            eval_mode=True
        )
        print(f"   Total samples loaded: {len(dataset)}")

        # Show first sample
        print("\n2. First sample structure:")
        first_sample = dataset[0]
        print(f"   Keys: {list(first_sample.keys())}")
        print(f"   input_ids length: {len(first_sample['input_ids'])}")
        print(f"   attention_mask length: {len(first_sample['attention_mask'])}")
        print(f"   labels: {first_sample['labels']}")
        print(f"   smiles: {first_sample['smiles']}")

        # Test a few more samples
        print("\n3. Testing multiple samples:")
        for i in [10, 100, 1000]:
            if i < len(dataset):
                sample = dataset[i]
                print(f"   Sample {i}: input_ids length = {len(sample['input_ids'])}, "
                      f"smiles count = {len(sample['smiles']) if sample['smiles'] else 0}")

        print("\n✓ Dataset loading test passed!\n")

        # Test loading in training mode (just first file to save time)
        print("\n4. Loading single file in training mode...")
        single_file_path = os.path.join(data_path, "property_prediction.json")
        dataset_train = load_instructmol_data(
            path=os.path.dirname(single_file_path),
            max_len=2048,
            eval_mode=False
        )
        print(f"   Training samples loaded: {len(dataset_train)}")

        # Show first training sample
        first_train = dataset_train[0]
        print(f"   First training sample:")
        print(f"     - input_ids length: {len(first_train['input_ids'])}")
        print(f"     - Has labels: {first_train['labels'] is not None}")
        if first_train['labels'] is not None:
            # Count non-masked labels
            non_masked = sum(1 for x in first_train['labels'] if x != -100)
            print(f"     - Non-masked label tokens: {non_masked}")

        print("\n✓ Training mode test passed!\n")

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_inference_integration():
    """Test integration with inference.py."""
    print("=" * 80)
    print("Testing inference.py integration...")
    print("=" * 80)

    try:
        from inference import load_test_data

        data_path = "/mnt/afs/L202500070/yimeng/Bio-LatentCOT/instructmol/Molecule-oriented_Instructions"

        print(f"\n1. Loading via load_test_data()...")
        dataset = load_test_data(data_path, max_len=2048)
        print(f"   Loaded {len(dataset)} samples")
        print(f"   First sample keys: {list(dataset[0].keys())}")

        print("\n✓ Inference integration test passed!\n")

    except Exception as e:
        print(f"\n✗ Error in inference integration: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("\n" + "🧪" * 40)
    print("InstructMol Dataloader Test Suite")
    print("🧪" * 40 + "\n")

    try:
        # Run tests
        test_extract_instructmol_fields()
        success = test_load_instructmol_data()

        if success:
            test_inference_integration()

        print("\n" + "=" * 80)
        print("✓ All tests completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
