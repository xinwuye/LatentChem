import pandas as pd
import random

INPUT_FILE = "./extracted/stage3/rna_stage3_with_gpt_responses.csv"
OUTPUT_FILE = "./extracted/stage3/gpt_sample_check.csv"
SAMPLE_SIZE = 50

def generate_sample():
    # 1. Load the dataset
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    # 2. Check if we have enough rows
    total_rows = len(df)
    if total_rows < SAMPLE_SIZE:
        print(f"Warning: File only has {total_rows} rows. Sampling all of them.")
        actual_sample_size = total_rows
    else:
        actual_sample_size = SAMPLE_SIZE

    # 3. Generate 50 random indices from the available rows
    # Using a seed ensures you can replicate this specific sample if needed
    random.seed(21) 
    sample_indices = random.sample(range(total_rows), actual_sample_size)

    # 4. Extract specific columns for those indices
    # We select 'input', 'output', and 'response'
    cols_to_keep = ['input', 'output', 'response']
    
    # Check if columns exist
    missing_cols = [c for c in cols_to_keep if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in CSV: {missing_cols}")
        return

    sample_df = df.iloc[sample_indices][cols_to_keep].copy()

    # 5. Rename 'response' to 'gpt summarized'
    sample_df = sample_df.rename(columns={'response': 'gpt summarized'})

    # 6. Save to new CSV
    sample_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Successfully saved {actual_sample_size} random samples to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_sample()