import pandas as pd
import re

# --- Paths ---
ORIGINAL_CSV = "./extracted/stage3/rna_stage3.csv"
TARGET_CSV = "./extracted/stage3/rna_stage3_last_paragraph.csv"
EXTRA_CHECK_FILE = "extra_check.txt"

def extract_last_n_paragraphs(text, n=2):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    # Split by double newlines
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', str(text)) if p.strip()]
    # Take the last 'n' paragraphs and join them back with double newlines
    return "\n\n".join(paragraphs[-n:]) if paragraphs else ""

def fix_extra_checks():
    # 1. Parse indices from extra_check.txt
    print(f"Parsing indices from {EXTRA_CHECK_FILE}...")
    with open(EXTRA_CHECK_FILE, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            print("extra_check.txt is empty. Nothing to do.")
            return
        entries = content.split('\n\n')
    
    # Extract the last integer from each entry (the index you appended earlier)
    failed_indices = []
    for entry in entries:
        parts = entry.strip().split()
        if parts:
            try:
                failed_indices.append(int(parts[-1]))
            except ValueError:
                continue

    print(f"Found {len(failed_indices)} indices to fix.")

    # 2. Load the CSVs
    print("Loading CSVs...")
    df_original = pd.read_csv(ORIGINAL_CSV)
    df_target = pd.read_csv(TARGET_CSV)

    # 3. Update only the failed rows
    # We use .at or .loc to modify the target dataframe in place
    count = 0
    for idx in failed_indices:
        if idx in df_original.index:
            # Get last 2 paragraphs from original 'output' column
            new_text = extract_last_n_paragraphs(df_original.at[idx, 'output'], n=2)
            # Update the 'last_paragraph' column in our target file
            df_target.at[idx, 'last_paragraph'] = new_text
            count += 1

    # 4. Save back to the same target CSV
    df_target.to_csv(TARGET_CSV, index=False)
    print(f"Successfully updated {count} rows in {TARGET_CSV} with 2-paragraph context.")

if __name__ == "__main__":
    fix_extra_checks()