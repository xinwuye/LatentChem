#!/usr/bin/env python3
"""
Test SELFIES to SMILES conversion for InstructMol dataset.
"""

import selfies as sf

def test_selfies_conversion():
    """Test SELFIES conversion with samples from InstructMol."""
    print("=" * 80)
    print("Testing SELFIES to SMILES Conversion")
    print("=" * 80)

    # Sample SELFIES from InstructMol dataset README
    test_cases = [
        {
            "name": "Simple aniline example",
            "selfies": "[N][C][=C][C][=C][C][=C][Ring1][=Branch1]",
            "expected_description": "aniline (aminobenzene)"
        },
        {
            "name": "Property prediction example",
            "selfies": "[O][=C][C][N][C][Branch1][Ring1][C][=O][C][=C][Ring1][#Branch1]",
            "expected_description": "cyclic molecule"
        },
        {
            "name": "Complex molecule",
            "selfies": "[C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][=Branch1][C][=O][O][C@H1][Branch2][Ring1][=Branch1][C][O][C][=Branch1][C][=O][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][C][O][P][=Branch1][C][=O][Branch1][C][O][O][C][C@@H1][Branch1][=Branch1][C][=Branch1][C][=O][O][N]",
            "expected_description": "phosphatidyl-L-serine"
        }
    ]

    print("\nTesting SELFIES to SMILES conversion:\n")

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"  SELFIES: {test['selfies'][:80]}..." if len(test['selfies']) > 80 else f"  SELFIES: {test['selfies']}")

        try:
            smiles = sf.decoder(test['selfies'])
            print(f"  SMILES:  {smiles}")
            print(f"  Expected: {test['expected_description']}")
            print(f"  ✓ Conversion successful!\n")
        except Exception as e:
            print(f"  ✗ Conversion failed: {e}\n")

    # Test with multiple molecules separated by '.'
    print("-" * 80)
    print("Testing multiple molecules (reaction):\n")

    reaction_input = "[O][=N+1][Branch1][C][O-1][C][=C][N][=C][Branch1][C][Cl][C][Branch1][C][I][=C][Ring1][Branch2].[Fe]"
    parts = reaction_input.split('.')

    print(f"Original: {reaction_input}")
    print(f"Split into {len(parts)} parts:\n")

    for i, part in enumerate(parts, 1):
        try:
            smiles = sf.decoder(part)
            print(f"  Part {i}: {part}")
            print(f"           → {smiles}\n")
        except Exception as e:
            print(f"  Part {i}: {part}")
            print(f"           ✗ Error: {e}\n")

    print("=" * 80)

if __name__ == "__main__":
    test_selfies_conversion()
