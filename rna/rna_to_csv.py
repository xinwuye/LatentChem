import pandas as pd
import ast
import os

base = 'extracted/stage3/'

input_file = base + 'rna.csv'
output_file = base + 'rna_flattened.csv'

print(f"Loading {input_file}...")

df = pd.read_csv(input_file)

df['extracted_sequences'] = df['extracted_sequences'].apply(ast.literal_eval)

print("Flattening sequences (1 id to many sequences)...")
df_flattened = df[['extracted_sequences']].explode('extracted_sequences')

df_flattened = df_flattened.reset_index()
df_flattened.columns = ['id', 'sequence']

df_flattened = df_flattened.dropna(subset=['sequence'])

print(f"Saving to {output_file}...")
df_flattened.to_csv(output_file, index=False)

print(f"Done! Created {len(df_flattened)} sequence rows from {len(df)} original records.")

print("\nFirst 5 rows of output:")
print(df_flattened.head())