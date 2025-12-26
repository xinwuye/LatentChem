import pandas as pd
import os

base = 'extracted/stage3/'

input_file = base + 'rna_stage3.csv'
output_file = base + 'rna_stage3.fasta'

print(f"Reading {input_file}...")

df = pd.read_csv(input_file, usecols=['extracted_sequences'])

print(f"Converting {len(df)} sequences to FASTA format...")

with open(output_file, 'w') as f:
    for idx, row in df.iterrows():
        seq_str = str(row['extracted_sequences'])
        
        clean_seq = seq_str.strip("[]'\" ")
        
        f.write(f">{idx}\n{clean_seq}\n")

print(f"Success! FASTA file saved at: {output_file}")