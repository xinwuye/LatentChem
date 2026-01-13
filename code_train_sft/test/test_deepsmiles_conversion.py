#!/usr/bin/env python3
"""
Quick test to verify DeepSMILES to SMILES conversion in dataloader.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import extract_instructmol_fields, deepsmiles_converter

def test_deepsmiles_conversion():
    """Test DeepSMILES conversion with a sample from InstructMol."""
    print("=" * 80)
    print("Testing DeepSMILES to SMILES Conversion")
    print("=" * 80)

    # Sample data with DeepSMILES format
    sample = {
        "instruction": "Please provide the energy separation between the highest occupied and lowest unoccupied molecular orbitals (HOMO-LUMO gap) of this molecule.",
        "input": "[O][=C][C][N][C][Branch1][Ring1][C][=O][C][=C][Ring1][#Branch1]",
        "output": 0.1913,
        "metadata": {
            "task": "property prediction",
            "split": "train"
        }
    }

    print("\nOriginal DeepSMILES input:")
    print(f"  {sample['input']}")

    # Test direct conversion
    print("\nDirect conversion test:")
    try:
        standard_smiles = deepsmiles_converter.decode(sample['input'])
        print(f"  Standard SMILES: {standard_smiles}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test through extract_instructmol_fields
    print("\nConverted through extract_instructmol_fields:")
    result = extract_instructmol_fields(sample, is_eval=False)
    print(f"  Converted SMILES: {result['input_smiles']}")
    print(f"  Query: {result['query'][:80]}...")
    print(f"  Label: {result['label']}")
    print(f"  Task: {result['task']}")

    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_deepsmiles_conversion()
