import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/inference')
from smi_ted_light.load import load_smi_ted
import pandas as pd
import numpy as np
import torch

# Chemistry
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.DataStructs import FingerprintSimilarity
from rdkit.DataStructs import TanimotoSimilarity

# function to canonicalize SMILES
def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized

# function to calculate pairwise Tanimoto similarity
def calculate_tanimoto_similarities(fps1, fps2):
    similarities = []
    for i in range(len(fps1)):
            sim = TanimotoSimilarity(fps1[i], fps2[i])
            similarities.append(sim)
    return similarities


# model_smi_ted = load_smi_ted(
#     folder='/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/inference/smi_ted_light',
#     ckpt_filename='/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted-Light_40.pt'
# )

# df_moses = pd.read_csv("/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/notebooks/data/moses_test.csv", nrows=1000)

# with torch.no_grad():
#     # print(type(df_moses['SMILES']))
#     # encode_embeddings = model_smi_ted.encode(df_moses['SMILES'], return_torch=True)
#     encode_embeddings = model_smi_ted.encode(["c1ccccc1"], return_torch=True)
    
# ============================
# 1. Molecule encoder (dummy)
# ============================
# class MockMoleculeEncoder(nn.Module):
#     def __init__(self, d_mol=384):
#         super().__init__()
#         self.d_mol = d_mol
        
#     def forward(self, smiles_ids):
#         # smile_ids: [B, L_smiles]
#         B, L = smiles_ids.shape
#         return torch.randn(B, L, self.d_mol).to(smiles_ids.device)


# ============================
# 2. Projector: d_mol → d_llm
# ============================
class QueryAttentionProjector(nn.Module):
    def __init__(self, input_dim=768, num_queries=4, output_dim=2560, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        self.query = nn.Parameter(torch.randn(1, num_queries, input_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):  # x: [1, 202, 768]
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # [1, 4, 768]
        attn_out, _ = self.attn(q, x, x)  # cross-attention: Q from query, K/V from x
        out = self.proj(attn_out)  # [1, 4, 2560]
        return out


# ============================
# 3. Multimodal Fusion with Qwen3
# ============================
class Qwen3MoleculeLLM(nn.Module):
    def __init__(self, 
                 qwen_model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B",
                 d_mol=202*768):
        super().__init__()

        # ---- load pretrained Qwen3 LLM ----
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        self.extra_tokens = ["<mol_start>", "<mol_end>"]

        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer.add_tokens(self.extra_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.start_id = self.tokenizer.convert_tokens_to_ids("<mol_start>")
        self.end_id   = self.tokenizer.convert_tokens_to_ids("<mol_end>")

        # 获取 LLm embedding 维度
        self.d_llm = self.llm.get_input_embeddings().weight.shape[1]

        # ---- molecule encoder and projector ----
        self.mol_encoder = load_smi_ted(
                folder='/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/inference/smi_ted_light',
                ckpt_filename='/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted-Light_40.pt'
            )
        self.projector = QueryAttentionProjector()

    def forward(self, smiles_ids, text_ids):
        """
        smiles_ids: [B, L_smiles]
        text_ids:   [B, L_text]
        """


        B = len(smiles_ids)


        # 1. Encode molecule
        with torch.no_grad():  # freeze encoder if desired
            mol_emb = self.mol_encoder.encode(smiles_ids,return_torch=True)   # [B, L_smiles, d_mol]
            # batch_size, seq_len, feature_dim = mol_emb.shape
            # mol_emb = mol_emb.view(batch_size, -1)

            
            mol_emb = mol_emb.to(self.llm.device)
            

        # 2. Project to LLM embedding space
        mol_emb_llm = self.projector(mol_emb)          # [B, L_smiles, d_llm]
        # mol_emb_llm = mol_emb_llm.view(1, 1, 2560)
                #设置特殊标记
        start_id = torch.tensor([self.start_id], device=mol_emb.device)
        end_id   = torch.tensor([self.end_id], device=mol_emb.device)
        embed = self.llm.get_input_embeddings()
        start_emb = embed(start_id).expand(B, 1, -1)
        end_emb   = embed(end_id).expand(B, 1, -1)
        #设置特殊标记
        
        #拼接特殊标记
        mol_emb_llm = torch.cat([
        start_emb,
        mol_emb_llm,
        end_emb
            ], dim=1)

                
        #拼接特殊标记

        # 3. Embed text tokens
        text_emb = self.llm.get_input_embeddings()(text_ids)  # [B, L_text, d_llm]

        # 4. Concatenate molecule embeddings BEFORE text embeddings
        fused_emb = torch.cat([mol_emb_llm, text_emb], dim=1)  # [B, L_total, d_llm]
        fused_emb = fused_emb.to(dtype=self.llm.dtype)  #对齐张量格式

        
        

        
        
        
        L_smiles = mol_emb_llm.shape[1]
        L_text = text_emb.shape[1]
        L_total = L_smiles + L_text

        # ---- build attention mask ----
        attn_mask = torch.ones(B, L_total, dtype=torch.long).to(fused_emb.device)

        # ---- build position ids ----
        pos_ids = torch.arange(0, L_total, device=fused_emb.device).unsqueeze(0)  # [1, L_total]

        # 5. Forward Qwen3 using custom embeddings
        outputs = self.llm(
            inputs_embeds=fused_emb,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            return_dict=True
        )

        return outputs.logits  # [B, L_total, vocab]


# ============================
# 4. Example usage
# ============================

if __name__ == "__main__":
    # 模拟输入
    # B = 2
    # L_smiles = 20
    # L_text = 40
    model = Qwen3MoleculeLLM(
    qwen_model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B",
    d_mol=202*768
    ).cuda()
    tokenizer = model.tokenizer

    texts = [
    "Please describe the functional groups of this molecule.",
    "Please describe the functional groups of this molecule  123  yes"
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    text_ids = enc["input_ids"].cuda()
    smiles_list = [
    "CC(=O)OC1=CC=CC=C1C(=O)O","CC(=O)OC1=CC=CC=C1C(=O)O"

    
    ]

    print("text_ids shape:", text_ids.shape)
    

    model = Qwen3MoleculeLLM().cuda()

    logits = model(smiles_list, text_ids)
    print("logits:", logits.shape)
