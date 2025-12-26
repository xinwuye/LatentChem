import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import sys
sys.path.append('/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/inference')
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
    def __init__(self, input_dim=768, num_queries=8, output_dim=4096, num_heads=8):
        """
        简化版本的查询注意力投影器（包含必要的归一化）
        修改：支持处理5维分子特征 [B, num_molecules, 1, L_mol, input_dim]
        """
        super().__init__()
        self.num_queries = num_queries
        self.output_dim = output_dim
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 可学习的查询向量
        self.query = nn.Parameter(torch.randn(1, num_queries, input_dim) * 0.02)
        
        # 查询向量归一化
        self.query_norm = nn.LayerNorm(input_dim)
        
        # 多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # 注意力后归一化
        self.post_attn_norm = nn.LayerNorm(input_dim)
        
        # 投影层
        self.proj = nn.Linear(input_dim, output_dim)
        
        # 投影前归一化
        self.pre_proj_norm = nn.LayerNorm(input_dim)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.query, gain=1.0)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.0)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        """
        前向传播，处理5维分子特征
        x: [B, num_molecules, 1, L_mol, input_dim]
        return: [B, num_molecules, num_queries, output_dim]
        """
        B, num_molecules, _, L_mol, d_input = x.shape
        
        # 重塑为 [B * num_molecules, L_mol, d_input] 用于批量处理
        # 首先压缩第2维（1维）
        x_squeezed = x.squeeze(2)  # [B, num_molecules, L_mol, d_input]
        
        # 然后重塑为 [B * num_molecules, L_mol, d_input]
        x_reshaped = x_squeezed.reshape(B * num_molecules, L_mol, d_input)
        
        # 输入归一化
        x_norm = self.input_norm(x_reshaped)
        
        # 准备查询向量
        q = self.query.expand(B * num_molecules, -1, -1)
        q = self.query_norm(q)
        
        # 注意力计算
        attn_out, _ = self.attn(q, x_norm, x_norm)
        
        # 残差连接 + 归一化
        residual = q + attn_out
        residual = self.post_attn_norm(residual)
        
        # 投影前归一化
        proj_input = self.pre_proj_norm(residual)
        
        # 最终投影
        out = self.proj(proj_input)
        
        # 重塑回原始形状 [B, num_molecules, num_queries, output_dim]
        out = out.reshape(B, num_molecules, self.num_queries, self.output_dim)
        
        return out


# ============================
# 2. 多模态融合模型 (兼容trl的SFTTrainer)
# ============================
class Qwen3MoleculeLLM(PreTrainedModel):
    def __init__(self, 
                 qwen_model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B",
                 d_mol=202*768):  # d_mol参数保留用于兼容性，实际未使用
        """
        分子-文本多模态大语言模型
        
        参数:
            qwen_model_name: Qwen基础模型路径
            d_mol: 分子特征维度（兼容性参数）
        """
        # 加载Qwen模型的配置文件
        config = PretrainedConfig.from_pretrained(qwen_model_name)
        super().__init__(config)

        # ---- 1. 加载预训练的Qwen LLM ----
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        
        # 添加分子特殊标记
        self.extra_tokens = ["<mol_start>", "<mol_end>"]
        self.tokenizer.add_tokens(self.extra_tokens)

        # 加载基础语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype=torch.float32,  # 使用半精度浮点数
            device_map="auto"           # 自动设备映射
        )
        
        # 调整词表大小以包含新添加的特殊标记
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 获取特殊标记的ID
        self.start_id = self.tokenizer.convert_tokens_to_ids("<mol_start>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<mol_end>")

        # 获取LLM的嵌入维度
        self.d_llm = self.model.get_input_embeddings().weight.shape[1]
        print(f"LLM嵌入维度: {self.d_llm}")

        # ---- 2. 分子编码器和投影器 ----
        # 加载预训练的分子编码器（SMI-TED）
        self.mol_encoder = load_smi_ted(
            folder='/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted/inference/smi_ted_light',
            ckpt_filename='/zengdaojian/zhangjia/BioLatent/smi-ted/smi-ted-Light_40.pt'
        )
        
        # 初始化投影器
        self.projector = QueryAttentionProjector()

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
        前向传播（训练和推理）
        
        注意: 必须通过kwargs传入smiles参数
        """
        # ============================
        # 1. 提取SMILES字符串
        # ============================
        smiles_list = kwargs.pop("smiles", None)
        if smiles_list is None:
            raise ValueError("训练/推理时必须提供smiles参数")

        B = len(smiles_list)  # 批次大小
        device = self.model.device  # 设备（CPU/GPU）

        # ============================
        # 2. 分子编码（编码器被冻结）
        # ============================
        with torch.no_grad():  # 冻结分子编码器，不计算梯度
            # 编码SMILES字符串为分子特征
            # 编码器输出形状为 [B, num_molecules, 1, L_mol, d_mol]
            mol_emb = self.mol_encoder.encode(
                smiles_list, return_torch=True
            ).to(device)  # [B, num_molecules, 1, L_mol, d_mol]
            
            print(f"分子编码器输出形状: {mol_emb.shape}")

        # 将分子特征投影到LLM嵌入空间
        mol_emb_llm = self.projector(mol_emb)  # [B, num_molecules, num_queries, d_llm]
        print(f"投影后分子特征形状: {mol_emb_llm.shape}")
        print(f"投影器输出维度: {mol_emb_llm.size(-1)}")
        print(f"LLM嵌入维度: {self.d_llm}")
        
        # 获取LLM的嵌入层
        embed = self.model.get_input_embeddings()
        
        # ============================
        # 3. 为每个分子添加开始和结束标记
        # ============================
        # 正确获取开始和结束标记的嵌入向量
        # 使用 token id 直接获取嵌入，确保形状正确
        with torch.no_grad():
            # 正确创建 token tensors
            start_token_tensor = torch.tensor([[self.start_id]], device=device)
            end_token_tensor = torch.tensor([[self.end_id]], device=device)
            
            # 获取嵌入向量
            start_emb = embed(start_token_tensor)  # [1, 1, d_llm]
            end_emb = embed(end_token_tensor)      # [1, 1, d_llm]
        
        print(f"开始标记嵌入形状: {start_emb.shape}")
        print(f"结束标记嵌入形状: {end_emb.shape}")
        
        # 为每个样本中的每个分子添加开始和结束标记
        mol_prefix_list = []
        for b in range(B):
            sample_molecules = []
            num_molecules = mol_emb_llm.size(1)  # 每个样本的分子数量
            
            for m in range(num_molecules):
                # 获取当前分子的特征 [num_queries, d_llm]
                molecule_feat = mol_emb_llm[b, m]  # [num_queries, d_llm]
                
                # 检查维度是否匹配
                if molecule_feat.size(-1) != start_emb.size(-1):
                    raise ValueError(f"分子特征维度 {molecule_feat.size(-1)} 与开始标记维度 {start_emb.size(-1)} 不匹配")
                
                # 添加开始和结束标记
                molecule_with_tags = torch.cat([
                    start_emb,  # [1, 1, d_llm]
                    molecule_feat.unsqueeze(0),  # [1, num_queries, d_llm]
                    end_emb     # [1, 1, d_llm]
                ], dim=1)  # [1, num_queries+2, d_llm]
                
                sample_molecules.append(molecule_with_tags)
            
            # 拼接当前样本中的所有分子
            if sample_molecules:
                sample_prefix = torch.cat(sample_molecules, dim=1)  # [1, total_mol_tokens, d_llm]
                mol_prefix_list.append(sample_prefix)
            else:
                # 如果没有分子，创建空的分子前缀
                empty_prefix = torch.zeros(1, 0, self.d_llm, device=device)
                mol_prefix_list.append(empty_prefix)
        
        # 将所有样本的分子前缀拼接
        mol_prefix = torch.cat(mol_prefix_list, dim=0)  # [B, total_mol_tokens, d_llm]
        L_mol = mol_prefix.size(1)  # 分子前缀的总长度
        
        print(f"分子前缀形状: {mol_prefix.shape}")

        # ============================
        # 4. 文本嵌入
        # ============================
        if inputs_embeds is None:
            # 如果未提供嵌入，通过token IDs计算
            text_emb = embed(input_ids)  # [B, L_text, d_llm]
        else:
            # 使用提供的嵌入
            text_emb = inputs_embeds

        print(f"文本嵌入形状: {text_emb.shape}")

        # ============================
        # 5. 融合分子和文本嵌入
        # ============================
        fused_embeds = torch.cat([mol_prefix, text_emb], dim=1).to(self.model.dtype)
        
        print(f"融合嵌入形状: {fused_embeds.shape}")

        # ============================
        # 6. 注意力掩码处理
        # ============================
        if attention_mask is not None:
            # 为分子部分创建全1的掩码
            mol_mask = torch.ones(B, L_mol, device=device, dtype=attention_mask.dtype)
            # 拼接分子掩码和文本掩码
            fused_attention_mask = torch.cat([mol_mask, attention_mask], dim=1)
        else:
            fused_attention_mask = None

        # ============================
        # 7. 位置ID处理
        # ============================
        if position_ids is not None:
            # 为分子部分创建连续位置ID
            mol_pos = torch.arange(L_mol, device=device).unsqueeze(0).expand(B, -1)
            # 文本部分的位置ID从L_mol开始
            fused_position_ids = torch.cat([mol_pos, position_ids + L_mol], dim=1)
        else:
            fused_position_ids = None

        # ============================
        # 8. 调用Qwen基础模型
        # ============================
        outputs = self.model(
            inputs_embeds=fused_embeds,           # 融合后的嵌入
            attention_mask=fused_attention_mask,  # 融合后的注意力掩码
            position_ids=fused_position_ids,      # 融合后的位置ID
            past_key_values=past_key_values,      # 用于加速生成的缓存
            use_cache=use_cache,                  # 是否使用缓存
            output_attentions=output_attentions,  # 是否输出注意力权重
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=True,                     # 返回字典格式
        )

        # 获取模型输出logits
        logits = outputs.logits  # [B, T, vocab_size]

        # ============================
        # 9. 计算损失（训练时使用）
        # ============================
        loss = None
        if labels is not None:
            # 调试信息：检查形状
            print(f"logits形状: {logits.shape}")
            print(f"labels形状: {labels.shape}")
            
            # 标准因果语言建模的shift方式
            shift_logits = logits[:, :-1, :].contiguous()  # 去掉最后一个token
            shift_labels = labels[:, 1:].contiguous()      # 去掉第一个token
            
            # 计算交叉熵损失
            # 注意：分子部分对应的标签应为-100（在数据处理时已设置）
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # 展平: [B*(T-1), vocab]
                shift_labels.view(-1),                         # 展平: [B*(T-1)]
                ignore_index=-100,                             # 忽略标签为-100的位置
            )

        # ============================
        # 10. 返回输出
        # ============================
        return CausalLMOutputWithPast(
            loss=loss,                        # 损失值（训练时）
            logits=logits,                    # 预测logits
            past_key_values=outputs.past_key_values,  # 缓存（用于生成）
            hidden_states=outputs.hidden_states,      # 隐藏状态
            attentions=outputs.attentions,            # 注意力权重
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
    ):
        """
        生成方法，支持处理多个分子的特征
        """
        device = input_ids.device
        B = input_ids.size(0)

        # =========================================================
        # 0. tokenizer 安全兜底
        # =========================================================
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # =========================================================
        # 1. 文本 embedding
        # =========================================================
        text_embeds = self.model.get_input_embeddings()(input_ids)
        # shape: [B, L_text, d_llm]

        L_text = text_embeds.size(1)

        # =========================================================
        # 2. 分子编码和投影
        # =========================================================
        with torch.no_grad():
            # 编码SMILES字符串为分子特征
            mol_emb = self.mol_encoder.encode(
                smiles_list, return_torch=True
            ).to(device)  # [B, num_molecules, 1, L_mol, d_mol]
            
            # 投影到LLM空间
            mol_embeds = self.projector(mol_emb)  # [B, num_molecules, num_queries, d_llm]
        
        # =========================================================
        # 3. 为每个分子添加开始和结束标记
        # =========================================================
        embed = self.model.get_input_embeddings()
        
        # 正确获取开始和结束标记的嵌入向量
        with torch.no_grad():
            start_token_tensor = torch.tensor([[self.start_id]], device=device)
            end_token_tensor = torch.tensor([[self.end_id]], device=device)
            
            start_emb = embed(start_token_tensor)  # [1, 1, d_llm]
            end_emb = embed(end_token_tensor)      # [1, 1, d_llm]
        
        # 为每个样本中的每个分子添加开始和结束标记
        mol_prefix_list = []
        for b in range(B):
            sample_molecules = []
            num_molecules = mol_embeds.size(1)  # 每个样本的分子数量
            
            for m in range(num_molecules):
                # 获取当前分子的特征 [num_queries, d_llm]
                molecule_feat = mol_embeds[b, m]  # [num_queries, d_llm]
                
                # 添加开始和结束标记
                molecule_with_tags = torch.cat([
                    start_emb,
                    molecule_feat.unsqueeze(0),
                    end_emb
                ], dim=1)  # [1, num_queries+2, d_llm]
                
                sample_molecules.append(molecule_with_tags)
            
            # 拼接当前样本中的所有分子
            if sample_molecules:
                sample_prefix = torch.cat(sample_molecules, dim=1)  # [1, total_mol_tokens, d_llm]
                mol_prefix_list.append(sample_prefix)
            else:
                # 如果没有分子，创建空的分子前缀
                empty_prefix = torch.zeros(1, 0, self.d_llm, device=device)
                mol_prefix_list.append(empty_prefix)
        
        # 将所有样本的分子前缀拼接
        mol_prefix = torch.cat(mol_prefix_list, dim=0)  # [B, total_mol_tokens, d_llm]
        L_mol = mol_prefix.size(1)
        
        # =========================================================
        # 4. 拼接 inputs_embeds
        # =========================================================
        inputs_embeds = torch.cat([mol_prefix, text_embeds], dim=1)
        # shape: [B, L_mol + L_text, d_llm]

        total_len = inputs_embeds.size(1)

        # =========================================================
        # 5. attention_mask处理
        # =========================================================
        if attention_mask is None:
            text_mask = torch.ones(B, L_text, device=device)
        else:
            text_mask = attention_mask.to(device)

        mol_mask = torch.ones(B, L_mol, device=device)
        fused_attention_mask = torch.cat([mol_mask, text_mask], dim=1).long()

        # =========================================================
        # 6. position_ids处理
        # =========================================================
        position_ids = torch.arange(
            total_len,
            device=device
        ).unsqueeze(0).expand(B, -1)

        # =========================================================
        # 7. 生成调用
        # =========================================================
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=fused_attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
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