import pandas as pd
import re

def extract_last_paragraph(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    
    # Split by double newlines (common paragraph separator)
    # Also handles multiple spaces/newlines
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', str(text)) if p.strip()]
    
    return paragraphs[-1] if paragraphs else ""

def process_csv(input_path, output_path, target_column='label'):
    # Load your data
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    if target_column not in df.columns:
        print(f"Error: Column '{target_column}' not found. Available: {df.columns.tolist()}")
        return

    # Extract the last paragraph
    print("Extracting last paragraphs...")
    df['last_paragraph'] = df[target_column].apply(extract_last_paragraph)
    
    # Save to new CSV (you can choose to save only the index and the result, or everything)
    # Here we save the original input and the new extracted column
    df.to_csv(output_path, index=False)
    print(f"Success! Saved to {output_path}")

# --- Execution ---
input_file = "./extracted/stage3/rna_stage3.csv"
output_file = "./extracted/stage3/rna_stage3_last_paragraph.csv"

process_csv(input_file, output_file, target_column='output')

# navigate to ./extracted/stage3/
# run python3 -c "import pandas as pd; df=pd.read_csv('rna_stage3_last_paragraph.csv'); print('\n\n'.join([f'{str(row.last_paragraph).strip()} {idx}' for idx, row in df.iterrows() if pd.notna(row.last_paragraph)]))" > final_outputs_with_indices.txt