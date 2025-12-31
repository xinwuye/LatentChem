import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import sys
import os
from config import ModelConfig

# 动态添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from smi_ted_light.loadnew import load_smi_ted
import torch.nn.functional as F
from transformers.generation.utils import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional

import torch
from typing import List, Optional


# ============================
# 1. 投影器：将分子特征映射到LLM空间
# ============================
class QueryAttentionProjector(nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_queries, 
                 output_dim, 
                 num_heads):
        """
        查询注意力投影器：将变长的分子 Token 序列压缩并映射到 LLM 空间
        """
        super().__init__()
        self.num_queries = num_queries
        self.output_dim = output_dim
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 可学习的查询向量 (Learned Queries)
        self.query = nn.Parameter(torch.zeros(1, num_queries, input_dim))
        
        # 多头注意力 (Cross-Attention)
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 注意力后归一化
        self.post_attn_norm = nn.LayerNorm(input_dim)
        
        # 投影层 (从分子维度映射到 LLM 维度)
        self.proj = nn.Linear(input_dim, output_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        # 查询向量通常使用正态分布初始化
        nn.init.normal_(self.query, std=0.02)
        # 线性层使用 Xavier 初始化
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x, key_padding_mask=None):
        """
        x: [B_total_mols, L_mol, input_dim]
        key_padding_mask: [B_total_mols, L_mol] (True 表示 Padding)
        """
        B_total = x.size(0)
        
        # 1. 输入归一化
        x_norm = self.input_norm(x)
        
        # 2. 准备查询向量 [B_total, num_queries, input_dim]
        q = self.query.expand(B_total, -1, -1)
        
        # 3. 交叉注意力：Queries 关注分子的 Encoder 输出
        attn_out, _ = self.attn(
            query=q, 
            key=x_norm, 
            value=x_norm, 
            key_padding_mask=key_padding_mask
        )
        
        # 4. 残差连接 + 归一化 (Post-LN)
        # 注意：这里 query (q) 充当了残差的骨架
        out = self.post_attn_norm(q + attn_out)
        
        # 5. 最终投影到 LLM 维度
        out = self.proj(out)
        
        return out


# ============================
# 2. 多模态融合模型 (兼容trl的SFTTrainer)
# ============================
class Qwen3MoleculeLLM(PreTrainedModel):
    def __init__(self, 
                 qwen_model_name,
                 mol_config,       # 🚨 必须传入配置字典
                 device_map=None):
        """
        分子-文本多模态大语言模型
        
        参数:
            qwen_model_name: Qwen基础模型路径
            mol_config: 包含 input_dim, num_queries, num_heads 等参数的字典
            device_map: 设备映射配置
        """
        # 加载Qwen模型的配置文件
        config = PretrainedConfig.from_pretrained(qwen_model_name)
        super().__init__(config)

        # 从 mol_config 解析参数
        self.num_queries = mol_config.get('num_queries', 128)
        self.mol_input_dim = mol_config.get('input_dim', 768)
        self.mol_num_heads = mol_config.get('num_heads', 8)
        self.smi_ted_folder = mol_config.get('smi_ted_folder', ModelConfig.DEFAULT_SMI_TED_FOLDER)
        self.smi_ted_ckpt = mol_config.get('smi_ted_ckpt', ModelConfig.DEFAULT_SMI_TED_CKPT)

        # ---- 1. 加载预训练的Qwen LLM ----
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        self.config._name_or_path = qwen_model_name

        # 添加分子特殊标记
        self.extra_tokens = ["<mol_start>", "<mol_end>"]
        self.tokenizer.add_tokens(self.extra_tokens)

        # 加载基础语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.float32,
            device_map=device_map
        )
        
        # 调整词表大小以包含新添加的特殊标记
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 获取特殊标记的ID
        self.start_id = self.tokenizer.convert_tokens_to_ids("<mol_start>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<mol_end>")

        # 获取LLM的嵌入维度
        self.d_llm = self.model.get_input_embeddings().weight.shape[1]

        # ---- 2. 分子编码器和投影器 ----
        # 加载预训练的分子编码器（SMI-TED）
        self.mol_encoder = load_smi_ted(
            folder=self.smi_ted_folder,
            ckpt_filename=self.smi_ted_ckpt
        )
        
        # 冻结分子编码器参数
        for param in self.mol_encoder.parameters():
            param.requires_grad = False
        self.mol_encoder.eval()
        
        # 初始化投影器，使用动态解析的参数
        self.projector = QueryAttentionProjector(
            input_dim=self.mol_input_dim,
            num_queries=self.num_queries,
            output_dim=self.d_llm,
            num_heads=self.mol_num_heads
        )
        # 确保投影器类型与基础模型一致
        self.projector.to(self.model.dtype)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        """
        重构后的前向传播：
        1. 批量处理变长分子。
        2. 动态拼接分子与文本（去Padding）。
        3. 重新全局对齐与Loss计算。
        """
        smiles_list = kwargs.pop("smiles", None)
        if smiles_list is None:
            raise ValueError("必须提供smiles参数")

        B = len(smiles_list)
        device = self.model.device

        # =========================================================
        # 1. 分子特征拉平与批量投影 (优化性能)
        # =========================================================
        with torch.no_grad():
            # mol_emb_nested: [[Tensor(L1, 768), Tensor(L2, 768)], [Tensor(L3, 768)]]
            mol_emb_nested = self.mol_encoder.encode(smiles_list)

        flat_mols = []
        mol_counts = []
        for sample_mols in mol_emb_nested:
            mol_counts.append(len(sample_mols))
            flat_mols.extend(sample_mols)

        if flat_mols:
            # 批量投影：将不同长度的分子特征 Padding 到 Batch 内最长长度
            max_L_mol = max(m.size(0) for m in flat_mols)
            padded_mols = torch.zeros(len(flat_mols), max_L_mol, self.mol_input_dim, device=device, dtype=self.model.dtype)
            mol_key_padding_mask = torch.ones(len(flat_mols), max_L_mol, device=device, dtype=torch.bool)
            
            for i, m in enumerate(flat_mols):
                curr_L = m.size(0)
                padded_mols[i, :curr_L] = m.to(device=device, dtype=self.model.dtype)
                mol_key_padding_mask[i, :curr_L] = False # False 为有效位置
            
            # 批量过投影器
            flat_feats_llm = self.projector(padded_mols, key_padding_mask=mol_key_padding_mask) # [Total_Mols, num_queries, d_llm]
        else:
            flat_feats_llm = []

        # 获取 LLM 嵌入层
        embed = self.model.get_input_embeddings()
        with torch.no_grad():
            start_emb = embed(torch.tensor([[self.start_id]], device=device)) # [1, 1, d_llm]
            end_emb = embed(torch.tensor([[self.end_id]], device=device))   # [1, 1, d_llm]

        # =========================================================
        # 2. 结构还原与变长融合 (去文本 Padding)
        # =========================================================
        if inputs_embeds is None:
            text_emb = embed(input_ids)
        else:
            text_emb = inputs_embeds
        text_emb = text_emb.to(dtype=self.model.dtype)

        fused_samples_list = []
        fused_labels_list = []
        cursor = 0

        for b in range(B):
            # 2.1 构造分子部分
            sample_mol_parts = []
            for _ in range(mol_counts[b]):
                m_feat = flat_feats_llm[cursor].unsqueeze(0) # [1, num_queries, d_llm]
                m_with_tags = torch.cat([start_emb, m_feat, end_emb], dim=1) # [1, num_queries+2, d_llm]
                sample_mol_parts.append(m_with_tags)
                cursor += 1
            
            mol_part = torch.cat(sample_mol_parts, dim=1) if sample_mol_parts else torch.zeros(1, 0, self.d_llm, device=device, dtype=self.model.dtype)

            # 2.2 提取真实文本内容 (通过 Mask 提取非 Padding 部分)
            if attention_mask is not None:
                # 无论左补齐还是右补齐，mask 为 1 的都是真实内容
                non_pad_indices = attention_mask[b].bool()
                t_emb = text_emb[b][non_pad_indices] # [real_len, d_llm]
                t_lab = labels[b][non_pad_indices] if labels is not None else None
            else:
                t_emb = text_emb[b]
                t_lab = labels[b] if labels is not None else None

            # 2.3 融合
            sample_fused = torch.cat([mol_part, t_emb.unsqueeze(0)], dim=1)
            
            # 2.4 处理 Labels
            if t_lab is not None:
                # 分子标记设为 -100 (不计算 Loss)
                mol_lab = torch.full((1, mol_part.size(1)), -100, device=device, dtype=t_lab.dtype)
                sample_lab = torch.cat([mol_lab, t_lab.unsqueeze(0)], dim=1)
                fused_labels_list.append(sample_lab)
            
            fused_samples_list.append(sample_fused)

        # =========================================================
        # 3. 全局重新对齐 (统一 Padding)
        # =========================================================
        max_fused_L = max(s.size(1) for s in fused_samples_list)
        final_embeds = torch.zeros(B, max_fused_L, self.d_llm, device=device, dtype=self.model.dtype)
        final_attn_mask = torch.zeros(B, max_fused_L, device=device, dtype=torch.long)
        final_labels = torch.full((B, max_fused_L), -100, device=device, dtype=torch.long) if labels is not None else None
        
        for b in range(B):
            curr_L = fused_samples_list[b].size(1)
            final_embeds[b, :curr_L] = fused_samples_list[b]
            final_attn_mask[b, :curr_L] = 1
            if final_labels is not None:
                final_labels[b, :curr_L] = fused_labels_list[b]

        # =========================================================
        # 4. 调用 LLM 并计算 Loss
        # =========================================================
        outputs = self.model(
            inputs_embeds=final_embeds,
            attention_mask=final_attn_mask,
            labels=final_labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        smiles_list: List[str],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ):
        """
        同步更新后的生成方法：
        1. 支持变长分子和批量投影。
        2. 推理阶段自动改用左补齐 (Left Padding) 以确保生成质量。
        """
        device = input_ids.device
        B = input_ids.size(0)

        # =========================================================
        # 1. 分子特征拉平与批量投影
        # =========================================================
        # mol_emb_nested: [[Tensor(L1, 768), ...], ...]
        mol_emb_nested = self.mol_encoder.encode(smiles_list)

        flat_mols = []
        mol_counts = []
        for sample_mols in mol_emb_nested:
            mol_counts.append(len(sample_mols))
            flat_mols.extend(sample_mols)

        if flat_mols:
            max_L_mol = max(m.size(0) for m in flat_mols)
            padded_mols = torch.zeros(len(flat_mols), max_L_mol, self.mol_input_dim, device=device, dtype=self.model.dtype)
            mol_key_padding_mask = torch.ones(len(flat_mols), max_L_mol, device=device, dtype=torch.bool)
            
            for i, m in enumerate(flat_mols):
                curr_L = m.size(0)
                padded_mols[i, :curr_L] = m.to(device=device, dtype=self.model.dtype)
                mol_key_padding_mask[i, :curr_L] = False
            
            flat_feats_llm = self.projector(padded_mols, key_padding_mask=mol_key_padding_mask)
        else:
            flat_feats_llm = []

        # 获取 LLM 嵌入层
        embed = self.model.get_input_embeddings()
        start_emb = embed(torch.tensor([[self.start_id]], device=device))
        end_emb = embed(torch.tensor([[self.end_id]], device=device))

        # =========================================================
        # 2. 结构还原与变长融合 (去文本 Padding)
        # =========================================================
        text_emb = embed(input_ids).to(dtype=self.model.dtype)
        fused_samples_list = []
        cursor = 0

        for b in range(B):
            # 2.1 构造分子部分
            sample_mol_parts = []
            for _ in range(mol_counts[b]):
                m_feat = flat_feats_llm[cursor].unsqueeze(0)
                m_with_tags = torch.cat([start_emb, m_feat, end_emb], dim=1)
                sample_mol_parts.append(m_with_tags)
                cursor += 1
            
            mol_part = torch.cat(sample_mol_parts, dim=1) if sample_mol_parts else torch.zeros(1, 0, self.d_llm, device=device, dtype=self.model.dtype)

            # 2.2 提取真实文本内容
            if attention_mask is not None:
                non_pad_indices = attention_mask[b].bool()
                t_emb = text_emb[b][non_pad_indices]
            else:
                t_emb = text_emb[b]

            # 2.3 融合
            sample_fused = torch.cat([mol_part, t_emb.unsqueeze(0)], dim=1)
            fused_samples_list.append(sample_fused)

        # =========================================================
        # 3. 推理专用：左补齐 (Left Padding)
        # =========================================================
        max_fused_L = max(s.size(1) for s in fused_samples_list)
        final_embeds = torch.zeros(B, max_fused_L, self.d_llm, device=device, dtype=self.model.dtype)
        final_attn_mask = torch.zeros(B, max_fused_L, device=device, dtype=torch.long)
        
        for b in range(B):
            curr_L = fused_samples_list[b].size(1)
            diff = max_fused_L - curr_L
            # 内容靠右放，左边留白 (0)
            final_embeds[b, diff:] = fused_samples_list[b]
            final_attn_mask[b, diff:] = 1

        # =========================================================
        # 4. 调用生成
        # =========================================================
        outputs = self.model.generate(
            inputs_embeds=final_embeds,
            attention_mask=final_attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **kwargs
        )

        return outputs


# ============================
# 3. 使用示例
# ============================
if __name__ == "__main__":
    # 初始化模型
    model = Qwen3MoleculeLLM(
        qwen_model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B",
    ).cuda()
    
    tokenizer = model.tokenizer

    # 示例文本
    texts = [
        "Please describe the functional groups of this molecule.",
        "Please describe the functional groups of this molecule."
    ]
    
    # 文本编码
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].cuda()
    attention_mask = enc["attention_mask"].cuda()

    # 示例SMILES列表（每个样本包含3个分子）
    smiles_list = [
        ["CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "C1=CC=C(C=C1)C=O"],  # 样本1的3个分子
        ["CC(=O)OC1=CC=CC=C1C(=O)O", "C1=CC=C(C=C1)C=O"]   # 样本2的3个分子
    ]
    
    # 示例标签（实际训练时会来自数据集）
    labels = [
        "This molecule contains carboxylic acid and ester functional groups.",
        "This molecule contains carboxylic acid and ester functional groups."
    ]

    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        smiles=smiles_list
    )
    
    # 输出logits形状
    print("模型输出logits形状:", outputs.logits.shape)