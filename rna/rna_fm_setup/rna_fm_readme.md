## Clone RNA-FM:
- git clone https://github.com/ml4bio/RNA-FM.git
- cd RNA-FM
## Replace environment 
replace the environmnet.yml in RNA-FM with the environment.yml in this folder, then, create environment 
- conda env create -f environment.yml
- conda activate RNA-FM

## Enter redevelopment folder 
- cd ./redevelop

## Get the .pth for RNA-FM 
Ensure mirror is on for faster download  
- export HF_ENDPOINT=https://hf-mirror.com
- huggingface-cli download cuhkaih/rnafm RNA-FM_pretrained.pth --local-dir .

## Move the .pth file to model cache to avoid inference-time redownload 
- mv RNA-FM_pretrained.pth ~/.cache/torch/hub/checkpoints/

Note: usually, you should move the .pth file to the location above. In the case that .pth still tries to download when you run inference, use the location indicated by the downloading message: 
"Downloading: "https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth" to <path>/RNA-FM_pretrained.pth"
Then, you should do: mv RNA-FM_pretrained.pth <path>

## Try running inference
python launch/predict.py \
    --config="pretrained/extract_embedding.yml" \
    --data_path="./data/examples/rna.fasta" \
    --save_dir="./results/biolatentcot" \
    --save_frequency 1 \
    --save_embeddings

## Replace with actual data in .fasta format, change configurations accordingly. 
For larger jobs with more GPUs, change the extract_embedding.yml file in RNA-FM/redevelop/data/pretrained. An example is provided in this folder. 