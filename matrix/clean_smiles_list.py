input_file = "/BioLatent/Bio-LatentCOT/refined/solubility/smiles.txt"
output_file = "/BioLatent/Bio-LatentCOT/refined/solubility/smiles.txt"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

if len(lines) % 10 != 0:
    print(f"警告：总行数 {len(lines)} 不能被 10 整除！")

unique_lines = [lines[i] for i in range(0, len(lines), 10)]

with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(unique_lines)