import pandas as pd 
import json
import re 
import os

# total data: 3330232
# RNA only: 2036888 -> ./extracted/rna.csv
# DNA only: 1044256 -> ./extra.cted/dna.cav

rna_list = [] 
protein_list = []
dna_list = [] 
other_tags = set()

file_path = 'stage2_train.jsonl'
pattern_rna = r"<rna>(.*?)<rna>"
pattern_dna = r"<dna>(.*?)<dna>"
pattern_protein = r"<protein>(.*?)<protein>"
other_pattern = r"<(.*?)>"

print("Starting processing...")

with open(file_path, 'r') as file:
    length = 3330232 
    for iter, line in enumerate(file):
        if iter % 100000 == 0: 
            print(f"Progress: {iter/length:.2%}")

        data = json.loads(line)
        user_input = data.get('input', '')

        others = re.findall(other_pattern, user_input) 
        for tag in others: 
            if tag not in ["protein", "rna", "dna", "/protein", "/rna", "/dna"]:
                other_tags.add(tag)

        rna_seq = re.findall(pattern_rna, user_input)
        dna_seq = re.findall(pattern_dna, user_input)
        protein_seq = re.findall(pattern_protein, user_input)

        types_present = sum([bool(rna_seq), bool(dna_seq), bool(protein_seq)])
        
        if types_present != 1:
            # Skip multi-molecule tasks (rna+dna, rna+protein, etc.) or empty ones
            continue 

        # Append to the correct list
        data['extracted_sequences'] = []
        if rna_seq:
            data['extracted_sequences'] = rna_seq
            rna_list.append(data)
        elif dna_seq:
            data['extracted_sequences'] = dna_seq
            dna_list.append(data)
        elif protein_list:
            data['extracted_sequences'] = protein_seq
            protein_list.append(data)

print(f"Extraction complete. Other tags found: {other_tags}")

output_dir = './extracted'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for name, data_list in [("rna", rna_list), ("dna", dna_list), ("protein", protein_list)]:
    if data_list:
        df = pd.DataFrame(data_list)
        save_path = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(save_path, index=False)
        print(f"Saved {len(df)} rows to {save_path}")
    else:
        print(f"No data found for {name}")

print(f"Process finished. Files saved in {output_dir}")