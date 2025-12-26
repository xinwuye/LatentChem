🛠️ Environment Installation
1️⃣ Clone the repository
git clone https://github.com/xinwuye/BioLatentCOT.git
cd your-repo


2️⃣ Create conda environment (recommended)
conda env create -f biolatent_environment.yml
conda activate vllm


📊 Data Preparation
1️⃣ Download dataset

ChemCOTDataset:
huggingface-cli download --resume-download --repo-type dataset OpenMol/ChemCoTDataset --local_dir your_file_path                                       

2️⃣Dataset repair,To fill in missing fields in the dataset, use single quotes (''). And we will not use the data in rxn for the time being.

python  xiufu.py

🧠 Model Download
Download pretrained model from Hugging Face
huggingface-cli download --resume-download\
  Qwen/Qwen3-8B-Base \
  --local-dir models/your-model

Download small molecule foundation model from Hugging Face
huggingface-cli download --resume-download\
  ibm-research/materials.smi-ted \
  --local-dir models/your-model

🏋️ Training

注意要将model_new.py中QueryAttentionProjector输出的维度，与train中collate_fn维度对齐，collate_fn为（QueryAttentionProjector输出的维度+2）*5

python train_sft.py \
  --mode train \
  --data_path chemcotbench-cot \
  --model_path ./qwen3_mol_sft_lora_results \
  --batch_size 2 \
  --max_seq_length 512 \
  --epochs 3





🔍 Inference
python train_lora.py \
  --mode inference \
  --model_path ./qwen3_mol_sft_lora_results


Output example：

INFO:__main__:Input SMILES: ['CC(=O)OC1=CC=CC=C1C(=O)O']
INFO:__main__:Input prompt: Please describe the functional groups of this molecule.
INFO:__main__:Model device: cuda:0
INFO:__main__:Input IDs device: cuda:0
INFO:__main__:Attention mask device: cuda:0
INFO:__main__:Generated response:  CN(C)COC1CC(=CCNCC(C(=CCNC(=CC(=CC(=CCCNC(=CCCN)c2CCCN(C)CCO)C(=CC(=CCC(=CC=Cc3CC=Cc3CC(=CC=Cc2CC=Cc4CC(=CC=CCc3CCCN(C)c4CCc4CC=C(c1CC(=CC(=CC(=CC(=CC(=CCCN(C=CCN(C)c2)C=Cc2c3CCO)CCc3CCO)CC(=CC(c2=CC=CC(C=Cc2CCc3=CC(=CC3CC=Cc2CC=CC(=CC=CC(=CC(=CC(=CC2CC(=CC=C(c2)Cc2CC3CC3CCC3CC=CC(c2)CC=C(c1CC
INFO:__main__:==================================================
