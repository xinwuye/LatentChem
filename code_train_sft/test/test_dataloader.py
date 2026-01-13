#!/usr/bin/env python
"""Quick test to verify dataloader pipeline works"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader import load_data
from config import ModelConfig

# Test with a small subset
DATA_PATH = ModelConfig.DEFAULT_DATA_PATH

print("=" * 60)
print("Testing Dataloader Pipeline")
print("=" * 60)

try:
    print(f"\nLoading data from: {DATA_PATH}")
    dataset = load_data(DATA_PATH, include_cot=True, max_len=512)  # Use smaller max_len for testing
    
    print(f"\n✓ Dataset loaded successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Column names: {dataset.column_names}")
    
    # Check first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n✓ First sample structure:")
        for key in sample.keys():
            if key == "smiles":
                print(f"  - {key}: {type(sample[key])} -> {sample[key]}")
            else:
                val = sample[key]
                if isinstance(val, list):
                    print(f"  - {key}: list of length {len(val)}")
                else:
                    print(f"  - {key}: {type(val)}")
        
        # Verify required fields
        required_fields = ["input_ids", "attention_mask", "labels", "smiles"]
        missing_fields = [f for f in required_fields if f not in sample]
        
        if missing_fields:
            print(f"\n✗ ERROR: Missing required fields: {missing_fields}")
            sys.exit(1)
        else:
            print(f"\n✓ All required fields present!")
            
        # Check types
        assert isinstance(sample["smiles"], list), "smiles should be a list"
        assert all(isinstance(s, str) for s in sample["smiles"]), "smiles should contain strings"
        assert isinstance(sample["input_ids"], list), "input_ids should be a list"
        assert isinstance(sample["labels"], list), "labels should be a list"
        
        print(f"✓ Field types are correct!")
        print(f"\n✓ Number of molecules in first sample: {len(sample['smiles'])}")
        
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

