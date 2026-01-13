#!/usr/bin/env python3
"""
Minimal test to verify DeepSMILES to SMILES conversion.
"""

import deepsmiles

def test_deepsmiles_conversion():
    """Test DeepSMILES conversion with a sample from InstructMol."""
    print("=" * 80)
    print("Testing DeepSMILES to SMILES Conversion")
    print("=" * 80)

    # Initialize converter
    converter = deepsmiles.Converter(rings=True, branches=True)

    # Sample DeepSMILES from InstructMol dataset
    deepsmiles_str = "[O][=C][C][N][C][Branch1][Ring1][C][=O][C][=C][Ring1][#Branch1]"

    print(f"\nOriginal DeepSMILES:")
    print(f"  {deepsmiles_str}")

    # Convert to standard SMILES
    try:
        standard_smiles = converter.decode(deepsmiles_str)
        print(f"\nConverted Standard SMILES:")
        print(f"  {standard_smiles}")
        print("\n✓ Conversion successful!")
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

    # Test with a simple example
    print("\n" + "-" * 80)
    print("Testing with a simpler DeepSMILES example:")
    print("-" * 80)

    # This is a known DeepSMILES example
    simple_deepsmiles = "CC))C"  # benzene ring in DeepSMILES
    print(f"\nDeepSMILES: {simple_deepsmiles}")
    try:
        simple_smiles = converter.decode(simple_deepsmiles)
        print(f"Standard SMILES: {simple_smiles}")
    except Exception as e:
        print(f"Error: {e}")

    # Test encoding a known SMILES
    print("\n" + "-" * 80)
    print("Testing encoding SMILES to DeepSMILES:")
    print("-" * 80)

    known_smiles = "c1ccccc1"
    print(f"\nStandard SMILES: {known_smiles}")
    try:
        encoded_deepsmiles = converter.encode(known_smiles)
        print(f"DeepSMILES: {encoded_deepsmiles}")
        decoded_back = converter.decode(encoded_deepsmiles)
        print(f"Decoded back: {decoded_back}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_deepsmiles_conversion()
