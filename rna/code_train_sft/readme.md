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

🏋️ Training sft

注意要将model_new.py中QueryAttentionProjector输出的维度，与train中collate_fn维度对齐，collate_fn为（QueryAttentionProjector输出的维度+2）*1

python train_sft_stage2.py \
  --mode train \
  --data_path chemcotbench-cot \
  --model_path ./qwen3_mol_sft_lora_results \
  --batch_size 2 \
  --max_seq_length 512 \
  --epochs 3


🏋️ Training sft_cot
python train_sft_stage2.py \
  --mode cotinue \
  --data_path chemcotbench-cot \
  --model_path ./qwen3_mol_sft_lora_results \
  --batch_size 2 \
  --max_seq_length 1024 \
  --epochs 3





🔍 Inference
python train_sft_stage2.py \
  --mode inference \
  --model_path ./qwen3_mol_sft_lora_results


Output example：

INFO:__main__:
==================================================
INFO:__main__:Input SMILES: ['CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl']
INFO:__main__:Input prompt: Modify the molecule CC1[NH2+]CCC1C(=O)Nc1cc(C(N)=O)ccc1Cl by adding a carboxyl.
INFO:__main__:Model device: cuda:0
INFO:__main__:Input IDs device: cuda:0
INFO:__main__:Attention mask device: cuda:0
tensor([[ 2691,   279,  1803,  ..., 19238,     8,  3706]], device='cuda:0')
generate_text:  Add the carbox to the end of the molecule. CC1[NH2CCC1[NH] to the C(=O] to the molecule. CC1[NH2CCC1CC(C(=OCC1[NH2CCC1[NH]CC1]C(=O]C(=OCC[NH]CO1[NH2C(=CCC1[NH2CCC2CC[NH]C(C(=O]C(=O]C(=O]C(=OCC1)C(O)C(=CC(O)C(O)C(C(=O]C(O)C(=O)C(=CC(C(=O)CCC(C(=CC1)C(=O)C(=O)N1)C(=O)C(=O)CCC1)C(=O)C1)C(C)CCC1C(=O]C1C(=CCC(C(=O)N(C(=O)C1)N(C(=O)C(O)C(=O)C(=CC(C)C(=OCCO)C(C(=CC2C(O)C2)CC[NH2O)N(C(=O2)C(C(=CC1)C1)C(C(=OCCO)CC2[NH2)C(=CC1)C(C(=O)C(=CC1C2)N(C)CC2)C(O)C(O)N(C)C(O)C(O)N2C(O)C2CC(C)CCO]CC1[NH2)C2C2C(O)C(O)C2C2CO(C)C2C(O)C1CC1C(O)C(C)N2C2C2CC2C1)C(=CC1CCO2C2CC2C2CO2CCO2CC(O)C(=CC1)C(O)C2)C2OCCO(O)N2C1CCO2CC2(C)C(O)C(O)CC2C(O)C1C(O)CC(O)CO2CC(C)C(C)C(C)C(COCCO2COCC2CO2CCC2C(O)C2C(=CC2C(O)C(C)C2CC1C1C12C2C2)CC1C(O)C2C(O)C(O)C2C2CCO2C2)C(O)C(O)C(C)CO2CC2C1CC(O)C2C1(O)C(C)CO(C)C(O)C(C(=O2C(O)C2COO2CO2C(O)C(O)C2C(C)C2C1CC1C2C(O)C(O)C(O)CC2CC2C1)C(O)C(O)C(O)C(O)C(O)CC2C(O)C(O)C(C)C2CO(CO2CO(C)C(O)C(C)CCO1C1CC2CCO1C(O)C2CO2CCC2C1C(O)CO(O)C(O)C(O)CO2CCOCC1C1C2C(O)C(O)C1C(O)CC2CO(O)C(O)C(C)C(C)C(O)C(O)C(O)C1CCOCCOCC1C(O)C(O)C2CCC(O)CC(O)C(O)C2C(O)C(O)CCO1OCCO2C(O)CO(O)C(O)C(O)CC(O)C(O)C(O)CC(O)C(O)CC1CO2CO(O)C(O)C(O)C2CCC1CC2C(O)C(O)C2C(O)C1C(O)C(C)C(O)CO(O)C(O)C(O)C2CCC2C(O)CC2CO2O2CC(O)C1CC2C(O)C(O)C(O)C(O)C(O)C(O)C2C(O)C(O)CC(O)C1CC(O)C(O)C2C(O)C(O)C(O)C(O)C1CCO(C)C(O)C(O)C(O)C1C2C(O)C(O)C(O)C(O)C2(O)C(O)C2C(O)CC(O)C(O)CCO2C(O)C(O)C(O)C(O)C(O)C(O)CO(C)C(O)C(O)C(O)C(O)C(O)CC
INFO:__main__:Generated response: O)C(C(=O]C(O)C(=O)C(=CC(C(=O)CCC(C(=CC1)C(=O)C(=O)N1)C(=O)C(=O)CCC1)C(=O)C1)C(C)CCC1C(=O]C1C(=CCC(C(=O)N(C(=O)C1)N(C(=O)C(O)C(=O)C(=CC(C)C(=OCCO)C(C(=CC2C(O)C2)CC[NH2O)N(C(=O2)C(C(=CC1)C1)C(C(=OCCO)CC2[NH2)C(=CC1)C(C(=O)C(=CC1C2)N(C)CC2)C(O)C(O)N(C)C(O)C(O)N2C(O)C2CC(C)CCO]CC1[NH2)C2C2C(O)C(O)C2C2CO(C)C2C(O)C1CC1C(O)C(C)N2C2C2CC2C1)C(=CC1CCO2C2CC2C2CO2CCO2CC(O)C(=CC1)C(O)C2)C2OCCO(O)N2C1CCO2CC2(C)C(O)C(O)CC2C(O)C1C(O)CC(O)CO2CC(C)C(C)C(C)C(COCCO2COCC2CO2CCC2C(O)C2C(=CC2C(O)C(C)C2CC1C1C12C2C2)CC1C(O)C2C(O)C(O)C2C2CCO2C2)C(O)C(O)C(C)CO2CC2C1CC(O)C2C1(O)C(C)CO(C)C(O)C(C(=O2C(O)C2COO2CO2C(O)C(O)C2C(C)C2C1CC1C2C(O)C(O)C(O)CC2CC2C1)C(O)C(O)C(O)C(O)C(O)CC2C(O)C(O)C(C)C2CO(CO2CO(C)C(O)C(C)CCO1C1CC2CCO1C(O)C2CO2CCC2C1C(O)CO(O)C(O)C(O)CO2CCOCC1C1C2C(O)C(O)C1C(O)CC2CO(O)C(O)C(C)C(C)C(O)C(O)C(O)C1CCOCCOCC1C(O)C(O)C2CCC(O)CC(O)C(O)C2C(O)C(O)CCO1OCCO2C(O)CO(O)C(O)C(O)CC(O)C(O)C(O)CC(O)C(O)CC1CO2CO(O)C(O)C(O)C2CCC1CC2C(O)C(O)C2C(O)C1C(O)C(C)C(O)CO(O)C(O)C(O)C2CCC2C(O)CC2CO2O2CC(O)C1CC2C(O)C(O)C(O)C(O)C(O)C(O)C2C(O)C(O)CC(O)C1CC(O)C(O)C2C(O)C(O)C(O)C(O)C1CCO(C)C(O)C(O)C(O)C1C2C(O)C(O)C(O)C(O)C2(O)C(O)C2C(O)CC(O)C(O)CCO2C(O)C(O)C(O)C(O)C(O)C(O)CO(C)C(O)C(O)C(O)C(O)C(O)CC
INFO:__main__:==================================================
