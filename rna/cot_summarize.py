import pandas as pd
from openai import OpenAI
import os
from tqdm import tqdm 

# --- Configuration ---
INPUT_CSV = "./extracted/stage3/rna_stage3.csv"
OUTPUT_CSV = "./extracted/stage3/rna_stage3_with_gpt_responses.csv"

client = OpenAI(
    base_url="http://34.13.73.248:3888/v1",
    api_key="sk-KEY"
)

def get_gpt_conclusion(text_to_summarize):
    try:
        # Note: Using your specific .responses.create syntax
        response = client.responses.create(
            model="gpt-5.1",
            temperature=0.0,
            input=(
                "You are extracting conclusions from scientific text.\n"
                "Rules:\n"
                "1. Ignore all reasoning, derivations, explanations, and intermediate steps.\n"
                "2. If a final conclusion is explicitly stated, extract it.\n"
                "3. If no explicit conclusion exists, infer the most likely conclusion.\n"
                "4. Output EXACTLY one sentence.\n"
                "5. Do NOT mention reasoning or uncertainty.\n\n"
                "Text:\n"
                f"{text_to_summarize}"
            )
        )
        return response.output_text.strip()
    except Exception as e:
        print(f"\nError processing row: {e}")
        return "ERROR_OR_TIMEOUT"

def safe_text(text):
    if not isinstance(text, str):
        return ""
    return text.encode("utf-8", errors="ignore").decode("utf-8")

def main():
    # 1. Load Original Data
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # 2. Check for Resume Point
    processed_count = 0
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        processed_count = len(df_existing)
        print(f"Resuming from row {processed_count}...")
    else:
        print("Starting fresh...")

    # 3. Only process the remaining rows
    # We slice the dataframe from processed_count to the end
    df_to_process = df.iloc[processed_count:]

    if df_to_process.empty:
        print("All rows already processed. Nothing to do!")
        return

    # 4. Process and Save
    print(f"Starting GPT API calls for {len(df_to_process)} remaining rows...")
    
    for i, row in tqdm(df_to_process.iterrows(), total=len(df_to_process)):
        result = get_gpt_conclusion(safe_text(row['output']))
        
        # Create a tiny dataframe for the single new row
        new_data = df.iloc[[i]].copy()
        new_data['response'] = [result]

        # Write to CSV: 
        # If it's the first row ever, write header. 
        # Otherwise, append (mode='a') and don't write header.
        header_needed = not os.path.exists(OUTPUT_CSV)
        new_data.to_csv(OUTPUT_CSV, mode='a', index=False, header=header_needed, encoding="utf-8")

    print(f"\nSuccess! All responses saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()