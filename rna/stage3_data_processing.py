import pandas as pd
import re
import os

# total: 8002 
# RNA: 2350
# DNA: 2831
# Protein: 2351

file_path = 'stage3.xlsx'
output_dir = './extracted/stage3'

pattern_rna = r"<rna>(.*?)<rna>"
pattern_dna = r"<dna>(.*?)<dna>"
pattern_protein = r"<protein>(.*?)<protein>"
other_pattern = r"<(.*?)>"

rna_list = [] 
protein_list = []
dna_list = [] 
other_tags = set()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Reading {file_path}...")

df_input = pd.read_excel(file_path)
total_rows = len(df_input)

print(f"Starting processing {total_rows} rows...")

for index, row in df_input.iterrows():
    if index % 1000 == 0:
        print(f"Progress: {index/total_rows:.2%}")

    data = row.to_dict()
    user_input = str(data.get('input', ''))

    others = re.findall(other_pattern, user_input) 
    for tag in others: 
        if tag not in ["protein", "rna", "dna", "/protein", "/rna", "/dna"]:
            other_tags.add(tag)

    rna_seqs = re.findall(pattern_rna, user_input)
    dna_seqs = re.findall(pattern_dna, user_input)
    protein_seqs = re.findall(pattern_protein, user_input)

    types_present = sum([bool(rna_seqs), bool(dna_seqs), bool(protein_seqs)])
    
    if types_present != 1:
        continue 

    if rna_seqs:
        data['extracted_sequences'] = rna_seqs[0]
        rna_list.append(data)
    elif dna_seqs:
        data['extracted_sequences'] = dna_seqs[0]
        dna_list.append(data)
    elif protein_seqs: # Fixed from protein_list to protein_seqs
        data['extracted_sequences'] = protein_seqs[0]
        protein_list.append(data)

print(f"Extraction complete. Other tags found: {other_tags}")

for name, data_list in [("rna", rna_list), ("dna", dna_list), ("protein", protein_list)]:
    if data_list:
        df_out = pd.DataFrame(data_list)
        save_path = os.path.join(output_dir, f'{name}_stage3.csv')
        df_out.to_csv(save_path, index=False)
        print(f"Saved {len(df_out)} rows to {save_path}")
    else:
        print(f"No data found for {name}")

print(f"Process finished. Files saved in {output_dir}")