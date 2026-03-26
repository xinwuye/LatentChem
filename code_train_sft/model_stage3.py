import torch
import torch.nn as nn
import math
from dataclasses import dataclass
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
from typing import Optional, List
import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import _disable_current_modes

@dataclass
class BioLatentCausalLMOutputWithPast(CausalLMOutputWithPast):
    ce_loss: Optional[torch.Tensor] = None
    bio_latent_loss: Optional[torch.Tensor] = None
    bio_latent_loss_scaled: Optional[torch.Tensor] = None
    bio_latent_active: Optional[bool] = None
    task_latent_loss: Optional[torch.Tensor] = None
    task_latent_loss_scaled: Optional[torch.Tensor] = None
    task_latent_active: Optional[bool] = None


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
# 1b. Stage 3 Memory Updater: 更新 BIO Token 的隐藏状态
# ============================
class BioTokenUpdater(nn.Module):
    def __init__(self, d_llm, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_llm, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_llm)
        self.ffn = nn.Sequential(
            nn.Linear(d_llm, d_llm * 2),
            nn.ReLU(),
            nn.Linear(d_llm * 2, d_llm)
        )
        self.norm_ffn = nn.LayerNorm(d_llm)

    def forward(self, bio_embeds, latent_states):
        """
        bio_embeds: [B, N_bio, d_llm]
        latent_states: [B, N_latent, d_llm]
        """
        # Cross-attention: Bio tokens attend to latent hidden states
        attn_out, _ = self.cross_attn(query=bio_embeds, key=latent_states, value=latent_states)
        bio_embeds = self.norm(bio_embeds + attn_out)
        
        # FFN
        ffn_out = self.ffn(bio_embeds)
        bio_embeds = self.norm_ffn(bio_embeds + ffn_out)
        return bio_embeds


class BioTokenUpdaterMulti(nn.Module):
    """
    Multi-expert variant of BioTokenUpdater: replaces the FFN MLP with 4 FFN "experts" and
    uses a softmax weighting gate to mix their outputs.
    """

    def __init__(self, d_llm: int, nhead: int = 8, n_experts: int = 4):
        super().__init__()
        self.n_experts = int(n_experts)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_llm, num_heads=nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_llm)

        self.gate = nn.Linear(d_llm, self.n_experts, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        hidden_dim = int(d_llm * 2)
        self.w1 = nn.Parameter(torch.empty(self.n_experts, hidden_dim, d_llm))
        self.b1 = nn.Parameter(torch.zeros(self.n_experts, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(self.n_experts, d_llm, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(self.n_experts, d_llm))

        for e in range(self.n_experts):
            nn.init.kaiming_uniform_(self.w1[e], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[e], a=math.sqrt(5))
        self.norm_ffn = nn.LayerNorm(d_llm)

    def forward(self, bio_embeds: torch.Tensor, latent_states: torch.Tensor) -> torch.Tensor:
        """
        bio_embeds: [B, N_bio, d_llm]
        latent_states: [B, N_latent, d_llm]
        """
        attn_out, _ = self.cross_attn(query=bio_embeds, key=latent_states, value=latent_states)
        bio_embeds = self.norm(bio_embeds + attn_out)

        weights = F.softmax(self.gate(bio_embeds.float()), dim=-1).to(dtype=bio_embeds.dtype)  # (B, N, E)

        hidden = torch.einsum("bnd,ehd->bneh", bio_embeds, self.w1) + self.b1.unsqueeze(0).unsqueeze(0)
        hidden = F.relu(hidden)

        out = torch.einsum("bneh,edh,bne->bnd", hidden, self.w2, weights) + torch.einsum("bne,ed->bnd", weights, self.b2)

        bio_embeds = self.norm_ffn(bio_embeds + out)
        return bio_embeds


# ============================
# 1b2. Hard gating: hard switch with straight-through gradients
# ============================
class HardSigmoidGate(nn.Linear):
    def __init__(self, d_llm: int, bias_init: float = 0.1, weight_init_std: float = 1e-3):
        super().__init__(int(d_llm), 1, bias=True)
        nn.init.normal_(self.weight, mean=0.0, std=float(weight_init_std))
        nn.init.constant_(self.bias, float(bias_init))

    def forward(self, states: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        states: [B, d]
        Returns gate: [B, 1, 1] using hard threshold in forward and sigmoid gradients in backward.
        """
        if states.dim() != 2:
            raise ValueError(
                f"{self.__class__.__name__} expects a 2D tensor [B, d], got shape={tuple(states.shape)}. "
                "Slice the token you want (e.g. `latent_states[:, -1, :]`) before calling."
            )
        logits = F.linear(states.float(), self.weight.float(), self.bias.float())
        prob = torch.sigmoid(logits)
        hard = (prob > 0.5).to(dtype=prob.dtype)
        gate = hard.detach() - prob.detach() + prob
        if out_dtype is None:
            out_dtype = states.dtype
        return gate.to(dtype=out_dtype).view(-1, 1, 1)


# Backward-compatible alias (older checkpoints/code used this name).
BioUpdaterGate = HardSigmoidGate


# ============================
# 1c. Bio Thinker: one-pass self-attn block for bio-latent tokens
# ============================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        self.d_model = int(d_model)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.d_model, 2).float() / self.d_model))
        self.register_buffer("_inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        position = torch.arange(L, device=x.device, dtype=self._inv_freq.dtype)
        sinusoid_inp = torch.einsum("i,j->ij", position, self._inv_freq.to(device=x.device))
        pe = torch.zeros(L, self.d_model, device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(sinusoid_inp).to(dtype=x.dtype)
        pe[:, 1::2] = torch.cos(sinusoid_inp).to(dtype=x.dtype)
        return pe.unsqueeze(0)


class BioThinker(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, dim_feedforward: Optional[int] = None):
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(d_model=d_model)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=(dim_feedforward if dim_feedforward is not None else d_model * 4),
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.pos(x)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        return self.layer(x, src_key_padding_mask=key_padding_mask)


class _TransformerEncoderLayerMultiFFN(nn.Module):
    """
    A minimal TransformerEncoderLayer (post-norm) with a softmax-weighted 4-expert FFN replacing the standard FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        n_experts: int = 4,
    ):
        super().__init__()
        self.n_experts = int(n_experts)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.gate = nn.Linear(d_model, self.n_experts, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self.dropout_ffn = float(dropout)
        self.w1 = nn.Parameter(torch.empty(self.n_experts, dim_feedforward, d_model))
        self.b1 = nn.Parameter(torch.zeros(self.n_experts, dim_feedforward))
        self.w2 = nn.Parameter(torch.empty(self.n_experts, d_model, dim_feedforward))
        self.b2 = nn.Parameter(torch.zeros(self.n_experts, d_model))
        for e in range(self.n_experts):
            nn.init.kaiming_uniform_(self.w1[e], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[e], a=math.sqrt(5))

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, *, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))

        weights = F.softmax(self.gate(x.float()), dim=-1).to(dtype=x.dtype)
        hidden = torch.einsum("bld,ehd->bleh", x, self.w1) + self.b1.unsqueeze(0).unsqueeze(0)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=self.dropout_ffn, training=self.training)

        out = torch.einsum("bleh,edh,ble->bld", hidden, self.w2, weights) + torch.einsum("ble,ed->bld", weights, self.b2)

        x = self.norm2(x + self.dropout2(out))
        return x


class BioThinkerMulti(nn.Module):
    """
    Multi-expert variant of BioThinker: replaces the Transformer FFN with 4 FFN experts mixed by a softmax gate.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, dim_feedforward: Optional[int] = None, n_experts: int = 4):
        super().__init__()
        self.pos = SinusoidalPositionalEncoding(d_model=d_model)
        self.layer = _TransformerEncoderLayerMultiFFN(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=(dim_feedforward if dim_feedforward is not None else d_model * 4),
            dropout=dropout,
            n_experts=n_experts,
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.pos(x)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        return self.layer(x, src_key_padding_mask=key_padding_mask)


class TaskThinker(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, int(d_model * hidden_mult))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(int(d_model * hidden_mult), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.delta(x)
        return x + y

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(self.norm(x)))))


class TaskThinkerMulti(nn.Module):
    """
    Multi-expert variant of TaskThinker: replaces the single MLP with 4 MLP "experts" and
    uses a softmax weighting gate to mix their outputs.
    """

    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0, n_experts: int = 4):
        super().__init__()
        self.n_experts = int(n_experts)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, self.n_experts, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        hidden_dim = int(d_model * hidden_mult)
        self.dropout = float(dropout)
        self.w1 = nn.Parameter(torch.empty(self.n_experts, hidden_dim, d_model))
        self.b1 = nn.Parameter(torch.zeros(self.n_experts, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(self.n_experts, d_model, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(self.n_experts, d_model))
        for e in range(self.n_experts):
            nn.init.kaiming_uniform_(self.w1[e], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w2[e], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.delta(x)
        return x + y

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        weights = F.softmax(self.gate(x_norm.float()), dim=-1).to(dtype=x_norm.dtype)

        hidden = torch.einsum("bld,ehd->bleh", x_norm, self.w1) + self.b1.unsqueeze(0).unsqueeze(0)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        out = torch.einsum("bleh,edh,ble->bld", hidden, self.w2, weights) + torch.einsum("ble,ed->bld", weights, self.b2)
        return out


# ============================
# 2. 多模态融合模型 (兼容trl的SFTTrainer)
# ============================
class Qwen3MoleculeLLM(PreTrainedModel):
    def __init__(
        self,
        qwen_model_name,
        mol_config,  # 🚨 必须传入配置字典
        device_map=None,
        is_coconut: bool = False,
        is_both_latent: bool = False,
        is_biothinker: bool = False,
        is_taskthinker: bool = False,
        is_bioupdater: bool = False,
        taskthinker_type: str = "mlp",
        is_biothinker_multi: bool = False,
        is_taskthinker_multi: bool = False,
        is_bioupdater_multi: bool = False,
        is_bioupdater_gating: bool = False,
        is_biothinker_gating: bool = False,
        is_taskthinker_gating: bool = False,
        mask_task_latent_steps: int = 0,
        mask_task_latent_noise_std: float = 1.0,
        shuffle_task_latents: bool = False,
        mask_task_latent_implementation: str = "noise",
        bio_latent_lambda: float = 0.0,
        bio_latent_alpha: float = 0.5,
        task_latent_lambda: float = 0.0,
        task_latent_alpha: float = 0.5,
        bio_thinker_dropout: float = 0.0,
        task_thinker_dropout: float = 0.0,
        max_cot_string_len: int = 2048,
        task_latent_max_steps: int = 10,
        torch_dtype=torch.bfloat16,
    ):
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
        self.is_coconut = bool(is_coconut)
        self.is_both_latent = bool(is_both_latent)
        self.is_biothinker = bool(is_biothinker)
        self.is_taskthinker = bool(is_taskthinker)
        self.is_bioupdater = bool(is_bioupdater)
        taskthinker_type_norm = str(taskthinker_type).strip().lower()
        if taskthinker_type_norm not in {"mlp", "identity"}:
            raise ValueError(f"Invalid taskthinker_type: {taskthinker_type!r}. Expected 'mlp' or 'identity'.")
        self.taskthinker_type = taskthinker_type_norm
        self.is_biothinker_multi = bool(is_biothinker_multi)
        self.is_taskthinker_multi = bool(is_taskthinker_multi)
        self.is_bioupdater_multi = bool(is_bioupdater_multi)
        self.is_bioupdater_gating = bool(is_bioupdater_gating)
        self.is_biothinker_gating = bool(is_biothinker_gating)
        self.is_taskthinker_gating = bool(is_taskthinker_gating)
        self.mask_task_latent_steps = int(mask_task_latent_steps)
        self.mask_task_latent_noise_std = float(mask_task_latent_noise_std)
        self.shuffle_task_latents = bool(shuffle_task_latents)
        self.mask_task_latent_implementation = str(mask_task_latent_implementation)
        self.bio_latent_lambda = float(bio_latent_lambda)
        self.bio_latent_alpha = float(bio_latent_alpha)
        self.task_latent_lambda = float(task_latent_lambda)
        self.task_latent_alpha = float(task_latent_alpha)
        self.bio_thinker_dropout = float(bio_thinker_dropout)
        self.task_thinker_dropout = float(task_thinker_dropout)
        self.max_cot_string_len = int(max_cot_string_len)
        self.task_latent_max_steps = int(task_latent_max_steps)

        # ---- 1. 加载预训练的Qwen LLM ----
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        self.config._name_or_path = qwen_model_name

        # 添加分子特殊标记
        self.extra_tokens = [
            "<mol_start>",
            "<mol_end>",
            "<latent>",
            "<start_latent>",
            "<end_latent>",
            "<bio_latent>",
            "<start_bio_latent>",
            "<end_bio_latent>",
        ]
        self.tokenizer.add_tokens(self.extra_tokens)

        # 加载基础语言模型
        self.model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            # IMPORTANT: fp32 doubles memory and will OOM easily for 8B models under GRPO.
            # Default to bf16 (good on Ampere/Hopper). Callers can override via `torch_dtype=...`.
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        # 调整词表大小以包含新添加的特殊标记
        # 注意：我们使用 max() 确保词表大小不小于原始 config 中的 vocab_size，
        # 这样可以保持与 vLLM (加载原始 config.json) 的兼容性，避免权重加载时的 AssertionError。
        new_vocab_size = max(len(self.tokenizer), self.model.config.vocab_size)
        self.model.resize_token_embeddings(new_vocab_size)
        
        # 获取特殊标记的ID
        self.start_id = self.tokenizer.convert_tokens_to_ids("<mol_start>")
        self.end_id = self.tokenizer.convert_tokens_to_ids("<mol_end>")
        self.latent_id = self.tokenizer.convert_tokens_to_ids("<latent>")
        self.start_latent_id = self.tokenizer.convert_tokens_to_ids("<start_latent>")
        self.end_latent_id = self.tokenizer.convert_tokens_to_ids("<end_latent>")
        self.bio_latent_id = self.tokenizer.convert_tokens_to_ids("<bio_latent>")
        self.start_bio_latent_id = self.tokenizer.convert_tokens_to_ids("<start_bio_latent>")
        self.end_bio_latent_id = self.tokenizer.convert_tokens_to_ids("<end_bio_latent>")

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

        # ---- Stage 3: Bio Token Updater ----
        if self.is_bioupdater_multi:
            self.bio_updater = BioTokenUpdaterMulti(d_llm=self.d_llm, nhead=self.mol_num_heads)
        else:
            self.bio_updater = BioTokenUpdater(d_llm=self.d_llm, nhead=self.mol_num_heads)
        self.bio_updater.to(self.model.dtype)
        self.bio_updater_gate: Optional[HardSigmoidGate] = None
        if self.is_bioupdater_gating:
            self.bio_updater_gate = HardSigmoidGate(self.d_llm, bias_init=0.1)
            self.bio_updater_gate.to(self.model.dtype)

        # ---- Stage 3: Bio Thinker (optional) ----
        if self.is_biothinker_multi:
            self.bio_thinker = BioThinkerMulti(
                d_model=self.d_llm,
                nhead=self.mol_num_heads,
                dropout=self.bio_thinker_dropout,
            )
        else:
            self.bio_thinker = BioThinker(
                d_model=self.d_llm,
                nhead=self.mol_num_heads,
                dropout=self.bio_thinker_dropout,
            )
        self.bio_thinker.to(self.model.dtype)
        self.bio_thinker_gate: Optional[HardSigmoidGate] = None
        if self.is_biothinker_gating:
            self.bio_thinker_gate = HardSigmoidGate(self.d_llm, bias_init=0.1)
            self.bio_thinker_gate.to(self.model.dtype)

        # ---- Stage 3: Task Thinker (optional) ----
        if self.is_taskthinker_multi:
            self.task_thinker = TaskThinkerMulti(d_model=self.d_llm, dropout=self.task_thinker_dropout)
        else:
            self.task_thinker = TaskThinker(d_model=self.d_llm, dropout=self.task_thinker_dropout)
        self.task_thinker.to(self.model.dtype)
        self.task_thinker_gate: Optional[HardSigmoidGate] = None
        if self.is_taskthinker_gating:
            self.task_thinker_gate = HardSigmoidGate(self.d_llm, bias_init=0.1)
            self.task_thinker_gate.to(self.model.dtype)

    # ---- Liger Kernel & Compatibility Helpers ----
    def _get_actual_llm(self):
        """Helper to get the underlying ForCausalLM model, even if wrapped in Peft."""
        llm = self.model
        if hasattr(llm, "base_model") and hasattr(llm.base_model, "model"):
            return llm.base_model.model
        return llm

    @property
    def norm(self):
        return self._get_actual_llm().model.norm

    @property
    def layers(self):
        return self._get_actual_llm().model.layers

    @property
    def embed_tokens(self):
        return self._get_actual_llm().model.embed_tokens

    @property
    def lm_head(self):
        return self._get_actual_llm().lm_head

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        """
        开启梯度检查点，转发给内部的语言模型。

        兼容 TRL/Transformers 的两种调用方式：
        - `gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(...))`
        - `gradient_checkpointing_enable(dict(...))`（将 dict 作为位置参数传入）
        """
        if not hasattr(self.model, "gradient_checkpointing_enable"):
            return

        if isinstance(gradient_checkpointing_kwargs, dict):
            merged = {**gradient_checkpointing_kwargs, **kwargs}
        elif gradient_checkpointing_kwargs is None:
            merged = dict(kwargs)
        else:
            # Unexpected positional argument type; ignore it.
            merged = dict(kwargs)

        try:
            # transformers 常见签名：gradient_checkpointing_enable(gradient_checkpointing_kwargs=...)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=merged)
        except TypeError:
            # 兼容少数实现：gradient_checkpointing_enable(**kwargs)
            self.model.gradient_checkpointing_enable(**merged)
            
    def gradient_checkpointing_disable(self):
        """
        关闭梯度检查点。
        """
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def _bioupdater_with_gating(self, bio_embeds: torch.Tensor, latent_states: torch.Tensor) -> torch.Tensor:
        refined = self.bio_updater(bio_embeds, latent_states)
        if not self.is_bioupdater_gating:
            return refined
        if self.bio_updater_gate is None:
            raise RuntimeError("is_bioupdater_gating=True but `bio_updater_gate` is not initialized.")

        gate = self.bio_updater_gate(latent_states[:, -1, :], out_dtype=refined.dtype)
        return refined * gate + bio_embeds * (1.0 - gate)

    def _taskthinker_with_gating(self, x: torch.Tensor) -> torch.Tensor:
        if self.taskthinker_type == "identity":
            return x
        if not self.is_taskthinker_gating:
            return self.task_thinker(x)
        if self.task_thinker_gate is None:
            raise RuntimeError("is_taskthinker_gating=True but `task_thinker_gate` is not initialized.")

        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(
                "TaskThinker gating expects a single-token embedding with shape [B, 1, d]. "
                f"Got x.shape={tuple(x.shape)}. Unsqueeze/slice before calling."
            )

        delta = self.task_thinker.delta(x)
        gate_in = x[:, 0, :]
        gate = self.task_thinker_gate(gate_in, out_dtype=delta.dtype)
        return x + delta * gate

    def _apply_latent_feedback(
        self,
        initial_embeds,
        attention_mask,
        latent_positions,
        bio_positions=None,
        refine_bio_tokens=True,
        task_thinker: Optional[nn.Module] = None,
        apply_task_thinker: bool = False,
    ):
        """
        Coconut 核心逻辑：潜空间迭代反馈。
        包含：更新 BIO Token 的隐藏状态 (Evidence Refinement)。
        """
        B = initial_embeds.shape[0]
        max_n_latents = max(len(l) for l in latent_positions) if latent_positions else 0
        
        if max_n_latents == 0:
            return initial_embeds

        llm = self._get_actual_llm()
        backbone = llm.model
        # has_lora_in_backbone = any(hasattr(m, "lora_A") and hasattr(m, "lora_B") for m in backbone.modules())
        # if not has_lora_in_backbone:
        #     raise RuntimeError("Expected LoRA layers in `llm.model` (backbone), but none was detected.")

        curr_embeds = initial_embeds
        bioupdater_gate_cache = None
        for pass_idx in range(max_n_latents):
            # 执行前向传播获取隐藏状态
            outputs = backbone(
                inputs_embeds=curr_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False
            )
            hidden_states = outputs.last_hidden_state
            
            new_embeds = curr_embeds.clone()

            # If gating is enabled, compute it once at the very beginning (pass 0) from the token *before* the first
            # latent position (for task-latent blocks, this is the <start_latent> token).
            if (
                pass_idx == 0
                and self.is_bioupdater_gating
                and refine_bio_tokens
                and bio_positions is not None
                and self.bio_updater_gate is not None
            ):
                gate_indices = [b for b in range(B) if bio_positions[b] and len(latent_positions[b]) > 0]
                if gate_indices:
                    anchor_states = []
                    for b in gate_indices:
                        first_pos = int(latent_positions[b][0])
                        anchor_pos = max(first_pos - 1, 0)
                        anchor_states.append(hidden_states[b, anchor_pos])
                    anchor = torch.stack(anchor_states, dim=0).to(dtype=self.model.dtype)  # (B_gate, d)

                    gate = self.bio_updater_gate(anchor, out_dtype=self.model.dtype)  # (B_gate, 1, 1)

                    bioupdater_gate_cache = torch.ones(B, 1, 1, device=anchor.device, dtype=self.model.dtype)
                    bioupdater_gate_cache[torch.tensor(gate_indices, device=anchor.device, dtype=torch.long)] = gate

            # --- Evidence Refinement (Memory Write) ---
            if refine_bio_tokens and bio_positions is not None:
                # 1. 识别当前 Pass 需要更新的 Batch 索引
                active_indices = [b for b in range(B) if bio_positions[b] and len(latent_positions[b]) > pass_idx]
                
                if active_indices:
                    # 2. 批量提取并补齐 BIO tokens [B_active, max_N_bio, d_llm]
                    bios = [curr_embeds[b, bio_positions[b]] for b in active_indices]
                    batched_bio = torch.nn.utils.rnn.pad_sequence(bios, batch_first=True)
                    batched_bio = batched_bio.to(self.model.dtype)
                    
                    # 3. 批量提取 Latent States [B_active, pass_idx + 1, d_llm]
                    lats = [hidden_states[b, latent_positions[b][:pass_idx + 1]] for b in active_indices]
                    batched_lat = torch.stack(lats)
                    batched_lat = batched_lat.to(self.model.dtype)
                    
                    # 4. 批量过 Cross-Attention 更新
                    refined = self.bio_updater(batched_bio, batched_lat)
                    if bioupdater_gate_cache is not None:
                        gates = bioupdater_gate_cache.index_select(
                            0, torch.tensor(active_indices, device=bioupdater_gate_cache.device, dtype=torch.long)
                        ).to(dtype=refined.dtype)
                        refined = refined * gates + batched_bio.to(dtype=refined.dtype) * (1.0 - gates)
                    
                    # 5. 将更新后的结果写回 (Scatter back)
                    for i, b in enumerate(active_indices):
                        new_embeds[b, bio_positions[b]] = refined[i, : len(bio_positions[b])].to(dtype=new_embeds.dtype)

            # 识别当前 Pass 需要更新 latent feedback 的 Batch 索引
            feedback_indices = [b for b in range(B) if len(latent_positions[b]) > pass_idx]
            if feedback_indices:
                device = initial_embeds.device
                pos_indices = torch.tensor([latent_positions[b][pass_idx] for b in feedback_indices], device=device)
                valid_mask = pos_indices > 0
                if valid_mask.any():
                    f_b_idx = torch.tensor(feedback_indices, device=device)[valid_mask]
                    f_p_idx = pos_indices[valid_mask]
                    # 批量注入：将前一个位置的输出作为当前位置的输入
                    new_embeds[f_b_idx, f_p_idx] = hidden_states[f_b_idx, f_p_idx - 1].to(dtype=new_embeds.dtype)

                    # --- TaskThinker refinement (optional, used for task-latent generation) ---
                    if apply_task_thinker and task_thinker is not None:
                        token = new_embeds[f_b_idx, f_p_idx].unsqueeze(1)  # [K, 1, d]
                        token = self._taskthinker_with_gating(token).squeeze(1)  # [K, d]
                        new_embeds[f_b_idx, f_p_idx] = token.to(dtype=new_embeds.dtype)

            curr_embeds = new_embeds
            
        return curr_embeds

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
        增强版前向传播：支持分子证据精炼与逆向干扰
        """
        smiles_list = kwargs.pop("smiles", None)
        do_perturb = kwargs.pop("do_perturb", False) # 是否执行逆向干扰 (Counterfactual perturbation)
        use_coconut = bool(kwargs.pop("is_coconut", self.is_coconut))
        use_both_latent = bool(kwargs.pop("is_both_latent", self.is_both_latent))
        use_biothinker_flag = bool(kwargs.pop("is_biothinker", self.is_biothinker))
        use_taskthinker_flag = bool(kwargs.pop("is_taskthinker", self.is_taskthinker))
        use_bioupdater_flag = bool(kwargs.pop("is_bioupdater", self.is_bioupdater))

        # If `is_both_latent=True`, always enable all three modules regardless of per-module flags.
        if use_both_latent:
            use_biothinker = True
            use_taskthinker = True
            use_bioupdater = True
        else:
            use_biothinker = use_biothinker_flag
            use_taskthinker = use_taskthinker_flag
            use_bioupdater = use_bioupdater_flag
        bio_latent_lambda = float(kwargs.pop("bio_latent_lambda", self.bio_latent_lambda))
        bio_latent_alpha = float(kwargs.pop("bio_latent_alpha", self.bio_latent_alpha))
        task_latent_lambda = float(kwargs.pop("task_latent_lambda", self.task_latent_lambda))
        task_latent_alpha = float(kwargs.pop("task_latent_alpha", self.task_latent_alpha))
        cot_len = kwargs.pop("cot_len", None)
        max_cot_string_len = int(kwargs.pop("max_cot_string_len", self.max_cot_string_len))

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
            start_bio_latent_emb = embed(torch.tensor([[self.start_bio_latent_id]], device=device))
            bio_latent_emb = embed(torch.tensor([[self.bio_latent_id]], device=device))
            end_bio_latent_emb = embed(torch.tensor([[self.end_bio_latent_id]], device=device))
            start_latent_emb = embed(torch.tensor([[self.start_latent_id]], device=device))
            latent_token_emb = embed(torch.tensor([[self.latent_id]], device=device))
            end_latent_emb = embed(torch.tensor([[self.end_latent_id]], device=device))

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
        bio_positions_list = [] # Stage 3: 记录每个 sample 中 bio token 的索引
        bio_latent_positions_list = []  # 记录每个 sample 的 bio latent token 索引（绝对位置）
        bio_latent_targets_list = []    # 每个 sample 的 v targets（长度 = #smiles），不参与梯度
        bio_latent_block_spans_list = []  # 每个 sample 的 bio-latent block span (start, end) 绝对位置
        bio_latent_anchor_pos_list = []  # 每个 sample 的 bio-latent anchor 位置（start_bio_latent 前一个 token）
        task_latent_positions_list = []  # 每个 sample 的 task latent token 索引（绝对位置，位于 <start_latent> 之后）
        prompt_spans_list = []  # 每个 sample 的 prompt span（融合序列绝对位置，左闭右开），用于 task-latent 对齐
        bio_thinker_visible_lens_list = []  # 每个 sample 中 bio_thinker 可见前缀长度（用于避免看到 response）
        cursor = 0

        for b in range(B):
            # 2.1 构造分子部分
            sample_mol_parts = []
            b_bio_indices = []
            current_mol_offset = 0
            b_targets = []

            for _ in range(mol_counts[b]):
                m_feat = flat_feats_llm[cursor].unsqueeze(0) # [1, num_queries, d_llm]
                # v: mean pooling of bio tokens (pre-concat), used as detached supervision target
                b_targets.append(m_feat.mean(dim=1).squeeze(0).detach())
                
                # --- Stage 3: Counterfactual Bio Latent Dropout ---
                if self.training and do_perturb:
                    # 随机干扰：Dropout, Shuffle, Noise
                    noise_type = torch.randint(0, 3, (1,)).item()
                    if noise_type == 0:
                        m_feat = torch.zeros_like(m_feat) # Dropout
                    elif noise_type == 1:
                        # Shuffle across the query tokens
                        idx = torch.randperm(m_feat.size(1))
                        m_feat = m_feat[:, idx, :] # Shuffle
                    elif noise_type == 2:
                        m_feat = m_feat + torch.randn_like(m_feat) * 0.05 # Noise

                m_with_tags = torch.cat([start_emb, m_feat, end_emb], dim=1) # [1, num_queries+2, d_llm]
                sample_mol_parts.append(m_with_tags)
                
                # 记录 bio tokens 的相对位置 (跳过 <mol_start>)
                start_query_pos = current_mol_offset + 1
                end_query_pos = start_query_pos + self.num_queries
                b_bio_indices.extend(range(start_query_pos, end_query_pos))
                current_mol_offset += (self.num_queries + 2)
                cursor += 1
            
            bio_positions_list.append(b_bio_indices)
            bio_latent_targets_list.append(b_targets)
            mol_part = torch.cat(sample_mol_parts, dim=1) if sample_mol_parts else torch.zeros(1, 0, self.d_llm, device=device, dtype=self.model.dtype)

            # 2.2 提取真实文本内容 (通过 Mask 提取非 Padding 部分)
            if attention_mask is not None:
                non_pad_indices = attention_mask[b].bool()
                t_emb_all = text_emb[b][non_pad_indices] # [real_len, d_llm]
                t_lab_all = labels[b][non_pad_indices] if labels is not None else None
                t_ids_all = input_ids[b][non_pad_indices] if input_ids is not None else None
            else:
                t_emb_all = text_emb[b]
                t_lab_all = labels[b] if labels is not None else None
                t_ids_all = input_ids[b] if input_ids is not None else None

            # Split text into prompt vs response using labels (-100 masks prompt).
            # For standard SFT: prompt | response(CoT+Answer+EOS)
            # For coconut SFT: prompt + <latent...> | response(remaining steps + answer)
            prompt_len = int(t_emb_all.size(0))
            if t_lab_all is not None:
                non_mask = (t_lab_all != -100)
                if non_mask.any():
                    prompt_len = int(non_mask.nonzero(as_tuple=True)[0][0].item())
            prompt_emb = t_emb_all[:prompt_len]
            resp_emb = t_emb_all[prompt_len:]

            prompt_span = None
            if prompt_len > 0:
                prompt_start = int(mol_part.size(1))
                prompt_end = int(mol_part.size(1) + prompt_len)
                prompt_span = (prompt_start, prompt_end)
            prompt_spans_list.append(prompt_span)

            # 2.2b 在 prompt 和 response 之间插入 Bio Latent tokens（数量 = smiles 个数）
            n_bio_latents = mol_counts[b]
            bio_latent_block = None
            bio_latent_block_len = 0
            bio_latent_positions = []
            if use_biothinker and n_bio_latents > 0:
                bio_latent_block_len = n_bio_latents + 2  # start + N + end
                bio_latents = bio_latent_emb.expand(1, n_bio_latents, -1)
                bio_latent_block = torch.cat([start_bio_latent_emb, bio_latents, end_bio_latent_emb], dim=1)

                base_len = int(mol_part.size(1) + prompt_emb.size(0))
                bio_latent_positions = list(range(base_len + 1, base_len + 1 + n_bio_latents))
                bio_latent_block_spans_list.append((base_len, base_len + bio_latent_block_len))
                bio_latent_anchor_pos_list.append(max(base_len - 1, 0))
            else:
                bio_latent_block_spans_list.append(None)
                bio_latent_anchor_pos_list.append(None)

            bio_latent_positions_list.append(bio_latent_positions)
            bio_thinker_visible_lens_list.append(int(mol_part.size(1) + prompt_emb.size(0) + bio_latent_block_len))

            # 2.2c 在 <end_bio_latent> 之后追加 Task Latent tokens（位于 prompt 与 response 之间）：
            # <start_latent> + N*<latent> + <end_latent>, N = ceil(len(cot)/max_cot_string_len * 4), capped at 4.
            n_task_latents = 0
            task_latent_block = None
            task_latent_block_len = 0
            task_latent_positions = []
            # NOTE: Coconut SFT already has <start_latent>/<latent>/<end_latent> in `input_ids`; avoid duplicating.
            if use_taskthinker and (not use_coconut):
                if cot_len is not None:
                    if isinstance(cot_len, torch.Tensor):
                        cot_len_b = int(cot_len[b].detach().cpu().item())
                    else:
                        cot_len_b = int(cot_len[b])
                    ratio = float(cot_len_b) / float(max_cot_string_len) if max_cot_string_len > 0 else 0.0
                    n_task_latents = int(min(4, max(0, math.ceil(ratio * 4.0))))

                task_latent_block_len = n_task_latents + 2  # start + N + end
                latent_placeholders = (
                    latent_token_emb.expand(1, n_task_latents, -1)
                    if n_task_latents > 0
                    else latent_token_emb[:, :0, :]
                )
                task_latent_block = torch.cat([start_latent_emb, latent_placeholders, end_latent_emb], dim=1)

                base_len = int(mol_part.size(1) + prompt_emb.size(0) + bio_latent_block_len)
                task_latent_positions = list(range(base_len + 1, base_len + 1 + n_task_latents))

            task_latent_positions_list.append(task_latent_positions)

            # 2.3 融合
            parts = [mol_part, prompt_emb.unsqueeze(0)]
            if bio_latent_block is not None:
                parts.append(bio_latent_block)
            if task_latent_block is not None:
                parts.append(task_latent_block)
            parts.append(resp_emb.unsqueeze(0))
            sample_fused = torch.cat(parts, dim=1)
            
            # 2.4 处理 Labels
            if t_lab_all is not None:
                # Mask any <latent> token IDs coming from text side (coconut SFT); inserted task-latents are handled below.
                t_lab_masked = t_lab_all.clone()
                if t_ids_all is not None:
                    is_latent = (t_ids_all == self.latent_id)
                    t_lab_masked[is_latent] = -100

                prompt_lab = t_lab_masked[:prompt_len]
                resp_lab = t_lab_masked[prompt_len:]

                # Molecule tokens: always masked
                mol_lab = torch.full((1, mol_part.size(1)), -100, device=device, dtype=t_lab_masked.dtype)
                lab_parts = [mol_lab, prompt_lab.unsqueeze(0)]

                # Bio-latent tokens: always masked
                if bio_latent_block_len > 0:
                    bio_lab = torch.full((1, bio_latent_block_len), -100, device=device, dtype=t_lab_masked.dtype)
                    lab_parts.append(bio_lab)

                # Task-latent tokens: <start_latent> and <latent>* are masked; <end_latent> is supervised.
                if task_latent_block_len > 0:
                    start_and_latents_lab = torch.full(
                        (1, 1 + n_task_latents), -100, device=device, dtype=t_lab_masked.dtype
                    )
                    end_latent_lab = torch.full(
                        (1, 1), int(self.end_latent_id), device=device, dtype=t_lab_masked.dtype
                    )
                    lab_parts.extend([start_and_latents_lab, end_latent_lab])

                # Response labels appended after the inserted latent blocks
                lab_parts.append(resp_lab.unsqueeze(0))

                sample_lab = torch.cat(lab_parts, dim=1)
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
        # 3b. Bio thinker: one-pass hidden thoughts for bio-latent tokens
        # =========================================================
        if use_biothinker and any(bio_latent_positions_list):
            # IMPORTANT: BioThinker is a bidirectional (encoder) block; it must NOT see teacher-forced response tokens.
            # We mask everything after the end of the bio-latent block as padding for BioThinker.
            bio_thinker_mask = torch.zeros_like(final_attn_mask)
            for b in range(B):
                curr_L = int(final_attn_mask[b].sum().item())
                vis_L = int(bio_thinker_visible_lens_list[b]) if b < len(bio_thinker_visible_lens_list) else curr_L
                vis_L = max(0, min(vis_L, curr_L))
                if vis_L > 0:
                    bio_thinker_mask[b, :vis_L] = 1

            thinker_out = self.bio_thinker(final_embeds, attention_mask=bio_thinker_mask)
            final_embeds_updated = final_embeds.clone()
            for b in range(B):
                positions = bio_latent_positions_list[b]
                if positions:
                    final_embeds_updated[b, positions] = thinker_out[b, positions].to(dtype=final_embeds_updated.dtype)
            if self.is_biothinker_gating and self.bio_thinker_gate is not None:
                active_indices = [b for b in range(B) if bio_latent_block_spans_list[b] is not None]
                if active_indices:
                    anchors = []
                    gate_inputs = []
                    for b in active_indices:
                        anchor_pos = bio_latent_anchor_pos_list[b]
                        if anchor_pos is None:
                            anchors.append(final_embeds_updated[b, 0])
                            gate_inputs.append(thinker_out[b, 0])
                        else:
                            anchor_idx = int(anchor_pos)
                            anchors.append(final_embeds_updated[b, anchor_idx])
                            gate_inputs.append(thinker_out[b, anchor_idx])
                    anchor_states = torch.stack(anchors, dim=0)  # (B_active, d)
                    gate_inputs_t = torch.stack(gate_inputs, dim=0)  # (B_active, d)
                    gates = self.bio_thinker_gate(
                        gate_inputs_t, out_dtype=final_embeds_updated.dtype
                    )  # (B_active, 1, 1)
                    for i, b in enumerate(active_indices):
                        span = bio_latent_block_spans_list[b]
                        if span is None:
                            continue
                        start, end = span
                        if end <= start:
                            continue
                        g = gates[i, 0, 0]
                        anchor = anchor_states[i].to(dtype=final_embeds_updated.dtype)
                        bio_block = final_embeds_updated[b, start:end].clone()
                        final_embeds_updated[b, start:end] = bio_block * g + anchor.unsqueeze(0) * (1.0 - g)
            final_embeds = final_embeds_updated

        # =========================================================
        # 4. Coconut 潜空间推理逻辑 (如果包含 <latent>)
        # =========================================================
        has_latent = (input_ids == self.latent_id).any().item()
        kwargs.pop("refine_bio_tokens", None)
        refine_bio_tokens = bool(use_bioupdater)

        model_input_embeds = final_embeds

        # 4a. Coconut latent-feedback refinement (from input_ids)
        if use_coconut and has_latent:
            latent_positions = []
            for b in range(B):
                mol_len_b = mol_counts[b] * (self.num_queries + 2)
                t_mask_b = attention_mask[b].bool() if attention_mask is not None else torch.ones_like(input_ids[b]).bool()
                rel_latent_indices = (input_ids[b][t_mask_b] == self.latent_id).nonzero(as_tuple=True)[0]
                latent_positions.append((rel_latent_indices + mol_len_b).tolist())

            model_input_embeds = self._apply_latent_feedback(
                model_input_embeds,
                final_attn_mask,
                latent_positions,
                bio_positions=bio_positions_list,
                refine_bio_tokens=refine_bio_tokens,
            )

        # 4b. Task latent generation after <end_bio_latent>
        if use_taskthinker and any(task_latent_positions_list):
            model_input_embeds = self._apply_latent_feedback(
                model_input_embeds,
                final_attn_mask,
                task_latent_positions_list,
                bio_positions=bio_positions_list,
                refine_bio_tokens=refine_bio_tokens,
                task_thinker=self.task_thinker,
                apply_task_thinker=True,
            )
        elif (not use_taskthinker) and refine_bio_tokens and (not (use_coconut and has_latent)):
            # Special case: BioUpdater enabled but TaskThinker disabled -> do a single memory update once,
            # without task-latent loop / refinement.
            active_indices = [
                b for b in range(B) if bio_positions_list[b] and int(bio_thinker_visible_lens_list[b]) > 0
            ]
            if active_indices:
                llm = self._get_actual_llm()
                backbone = llm.model

                update_mask = torch.zeros_like(final_attn_mask)
                anchor_pos_list: list[int] = []
                for b in active_indices:
                    curr_L = int(final_attn_mask[b].sum().item())
                    vis_L = int(bio_thinker_visible_lens_list[b])
                    vis_L = max(0, min(vis_L, curr_L))
                    if vis_L <= 0:
                        anchor_pos_list.append(0)
                        continue
                    update_mask[b, :vis_L] = 1
                    anchor_pos_list.append(vis_L - 1)

                embeds_in = model_input_embeds
                out = backbone(
                    inputs_embeds=embeds_in,
                    attention_mask=update_mask,
                    return_dict=True,
                    use_cache=False,
                )
                hidden_states = out.last_hidden_state  # (B, L, d)

                bios = [embeds_in[b, bio_positions_list[b]] for b in active_indices]
                batched_bio = torch.nn.utils.rnn.pad_sequence(bios, batch_first=True)
                lats = [hidden_states[b, anchor_pos_list[i]].unsqueeze(0) for i, b in enumerate(active_indices)]
                batched_lat = torch.stack(lats, dim=0)  # (B_active, 1, d)

                refined = self._bioupdater_with_gating(
                    batched_bio.to(self.model.dtype), batched_lat.to(self.model.dtype)
                )

                embeds_out = embeds_in.clone()
                for i, b in enumerate(active_indices):
                    embeds_out[b, bio_positions_list[b]] = refined[i, : len(bio_positions_list[b])].to(
                        dtype=embeds_out.dtype
                    )
                model_input_embeds = embeds_out

        # 4c. Final forward
        outputs = self.model(
            inputs_embeds=model_input_embeds,
            attention_mask=final_attn_mask,
            labels=final_labels,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # =========================================================
        # 5. Bio latent alignment loss (optional)
        # Loss = avg_i max(0, alpha - cos(v_i, mu_i)), where v_i is detached.
        # =========================================================
        ce_loss = outputs.loss
        if ce_loss is None:
            # GRPO log-prob computation (and some inference paths) call `forward()` without labels.
            # In that case HF returns `loss=None`; skip all auxiliary loss bookkeeping and only return logits.
            return BioLatentCausalLMOutputWithPast(
                loss=None,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                ce_loss=None,
                bio_latent_loss=None,
                bio_latent_loss_scaled=None,
                bio_latent_active=False,
                task_latent_loss=None,
                task_latent_loss_scaled=None,
                task_latent_active=False,
            )

        total_loss = ce_loss

        bio_latent_loss = ce_loss.new_tensor(0.0)
        bio_latent_loss_scaled = ce_loss.new_tensor(0.0)
        bio_latent_active = False

        task_latent_loss = ce_loss.new_tensor(0.0)
        task_latent_loss_scaled = ce_loss.new_tensor(0.0)
        task_latent_active = False

        if use_biothinker and bio_latent_lambda > 0.0 and any(bio_latent_positions_list):
            sample_losses = []
            for b in range(B):
                positions = bio_latent_positions_list[b]
                if not positions:
                    continue
                targets = bio_latent_targets_list[b]
                if len(targets) != len(positions):
                    raise ValueError(
                        f"bio_latent target/position mismatch: targets={len(targets)} positions={len(positions)}"
                    )
                mu = model_input_embeds[b, positions].float()
                v = torch.stack(targets, dim=0).to(device=mu.device).float().detach()
                cos_sim = F.cosine_similarity(v, mu, dim=-1)
                alpha = mu.new_tensor(bio_latent_alpha)
                sample_losses.append(F.relu(alpha - cos_sim).mean())

            if sample_losses:
                bio_latent_loss = torch.stack(sample_losses).mean()
                bio_latent_loss_scaled = (bio_latent_lambda * bio_latent_loss).to(dtype=total_loss.dtype)
                total_loss = total_loss + bio_latent_loss_scaled
                bio_latent_active = True

        # =========================================================
        # 6. Task latent prompt-alignment loss (optional)
        # Loss = avg_i max(0, alpha - cos(v_prompt_mean, mu_task_i)), where v is detached.
        # =========================================================
        if use_taskthinker and task_latent_lambda > 0.0 and any(task_latent_positions_list):
            sample_losses = []
            for b in range(B):
                positions = task_latent_positions_list[b]
                if not positions:
                    continue
                prompt_span = prompt_spans_list[b]
                if prompt_span is None:
                    continue

                mu = model_input_embeds[b, positions].float()
                p_start, p_end = prompt_span
                if p_end <= p_start:
                    continue
                v = model_input_embeds[b, p_start:p_end].float().detach().mean(dim=0)
                cos_sim = F.cosine_similarity(mu, v.unsqueeze(0).expand_as(mu), dim=-1)
                alpha = mu.new_tensor(task_latent_alpha)
                sample_losses.append(F.relu(alpha - cos_sim).mean())

            if sample_losses:
                task_latent_loss = torch.stack(sample_losses).mean()
                task_latent_loss_scaled = (task_latent_lambda * task_latent_loss).to(dtype=total_loss.dtype)
                total_loss = total_loss + task_latent_loss_scaled
                task_latent_active = True

        out = BioLatentCausalLMOutputWithPast(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            ce_loss=ce_loss,
            bio_latent_loss=bio_latent_loss,
            bio_latent_loss_scaled=bio_latent_loss_scaled,
            bio_latent_active=bio_latent_active,
            task_latent_loss=task_latent_loss,
            task_latent_loss_scaled=task_latent_loss_scaled,
            task_latent_active=task_latent_active,
        )
        return out

    def get_prompt_embeddings(
        self,
        smiles_list: List[List[str]],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        refine_bio_tokens: bool = True,
        corrupt_task_latents: Optional[torch.Tensor] = None,
        corrupt_task_latent_noise_std: float = 0.0,
    ):
        """
        Build fused prompt embeddings for generation.

        This mirrors `generate()` up to (and including) the Coconut latent-feedback refinement.
        It returns:
        - `prompt_embeds`: (B, L, d_llm) fused embeddings (left-padded)
        - `prompt_attn_mask`: (B, L) attention mask aligned to `prompt_embeds`

        This is used by vLLM generation paths that accept `prompt_embeds`.
        """
        device = input_ids.device
        B = input_ids.size(0)
        use_both_latent = bool(self.is_both_latent)
        if use_both_latent:
            use_biothinker = True
            use_taskthinker = True
            use_bioupdater = True
        else:
            use_biothinker = bool(self.is_biothinker)
            use_taskthinker = bool(self.is_taskthinker)
            use_bioupdater = bool(self.is_bioupdater)
        use_coconut = bool(self.is_coconut)

        self.total_inference_flops = 0
        with FlopCounterMode(display=False) as flop_counter:
            # Corruption flags are per-sample (B,). When enabled, task latent embeddings are generated normally, then
            # replaced with "no-information" vectors before answer generation.
            if corrupt_task_latents is None:
                corrupt_flags = [False for _ in range(B)]
            elif isinstance(corrupt_task_latents, torch.Tensor):
                corrupt_flags = [bool(x) for x in corrupt_task_latents.detach().cpu().tolist()]
            else:
                corrupt_flags = [bool(x) for x in corrupt_task_latents]
            if len(corrupt_flags) != B:
                raise ValueError(f"corrupt_task_latents length mismatch: got {len(corrupt_flags)} expected {B}")

            self._last_task_latent_counts = [0 for _ in range(B)]
            # If `is_both_latent=True`, BioUpdater is always enabled. Otherwise it's controlled by `is_bioupdater`.
            refine_bio_tokens = True if use_both_latent else (bool(refine_bio_tokens) and use_bioupdater)

            # NOTE: This function is used both for generation (call under `torch.no_grad()` / `torch.inference_mode()`)
            # and for GRPO log-prob computation (needs gradients for projector/bio_updater/bio_thinker/task_thinker).

            # =========================================================
            # 1. Molecule features: flatten + batch projection
            # =========================================================
            with torch.no_grad():
                mol_emb_nested = self.mol_encoder.encode(smiles_list)

            flat_mols = []
            mol_counts = []
            for sample_mols in mol_emb_nested:
                mol_counts.append(len(sample_mols))
                flat_mols.extend(sample_mols)

            if flat_mols:
                max_L_mol = max(m.size(0) for m in flat_mols)
                padded_mols = torch.zeros(
                    len(flat_mols), max_L_mol, self.mol_input_dim, device=device, dtype=self.model.dtype
                )
                mol_key_padding_mask = torch.ones(len(flat_mols), max_L_mol, device=device, dtype=torch.bool)

                for i, m in enumerate(flat_mols):
                    curr_L = m.size(0)
                    padded_mols[i, :curr_L] = m.to(device=device, dtype=self.model.dtype)
                    mol_key_padding_mask[i, :curr_L] = False

                flat_feats_llm = self.projector(padded_mols, key_padding_mask=mol_key_padding_mask)
            else:
                flat_feats_llm = []

            # LLM embedding layer
            embed = self.model.get_input_embeddings()
            start_emb = embed(torch.tensor([[self.start_id]], device=device))
            end_emb = embed(torch.tensor([[self.end_id]], device=device))
            start_bio_latent_emb = embed(torch.tensor([[self.start_bio_latent_id]], device=device))
            bio_latent_emb = embed(torch.tensor([[self.bio_latent_id]], device=device))
            end_bio_latent_emb = embed(torch.tensor([[self.end_bio_latent_id]], device=device))
            start_latent_emb = embed(torch.tensor([[self.start_latent_id]], device=device))
            end_latent_emb = embed(torch.tensor([[self.end_latent_id]], device=device))

            # =========================================================
            # 2. Reconstruct + fuse variable-length (strip text padding)
            # =========================================================
            text_emb = embed(input_ids).to(dtype=self.model.dtype)
            fused_samples_list = []
            bio_positions_list = []
            bio_latent_positions_list = []
            bio_latent_block_spans_list = []
            bio_latent_anchor_pos_list = []
            cursor = 0

            for b in range(B):
                sample_mol_parts = []
                b_bio_indices = []
                current_mol_offset = 0
                for _ in range(mol_counts[b]):
                    m_feat = flat_feats_llm[cursor].unsqueeze(0)
                    m_with_tags = torch.cat([start_emb, m_feat, end_emb], dim=1)
                    sample_mol_parts.append(m_with_tags)

                    start_query_pos = current_mol_offset + 1
                    end_query_pos = start_query_pos + self.num_queries
                    b_bio_indices.extend(range(start_query_pos, end_query_pos))
                    current_mol_offset += (self.num_queries + 2)
                    cursor += 1

                bio_positions_list.append(b_bio_indices)
                mol_part = (
                    torch.cat(sample_mol_parts, dim=1)
                    if sample_mol_parts
                    else torch.zeros(1, 0, self.d_llm, device=device, dtype=self.model.dtype)
                )

                if attention_mask is not None:
                    non_pad_indices = attention_mask[b].bool()
                    t_emb = text_emb[b][non_pad_indices]
                else:
                    t_emb = text_emb[b]

                n_bio_latents = mol_counts[b]
                bio_latent_block = None
                bio_latent_positions = []
                if use_biothinker and n_bio_latents > 0:
                    bio_latents = bio_latent_emb.expand(1, n_bio_latents, -1)
                    bio_latent_block = torch.cat([start_bio_latent_emb, bio_latents, end_bio_latent_emb], dim=1)

                    base_len = mol_part.size(1) + t_emb.size(0)
                    bio_latent_positions = list(range(base_len + 1, base_len + 1 + n_bio_latents))
                    bio_latent_block_spans_list.append((int(base_len), int(base_len + n_bio_latents + 2)))
                    bio_latent_anchor_pos_list.append(max(int(base_len) - 1, 0))
                else:
                    bio_latent_block_spans_list.append(None)
                    bio_latent_anchor_pos_list.append(None)

                bio_latent_positions_list.append(bio_latent_positions)

                parts = [mol_part, t_emb.unsqueeze(0)]
                if bio_latent_block is not None:
                    parts.append(bio_latent_block)
                # NOTE: Coconut mode already includes <start_latent>/<latent>/<end_latent> in `input_ids`.
                if use_taskthinker and (not use_coconut):
                    parts.append(start_latent_emb)
                fused_samples_list.append(torch.cat(parts, dim=1))

            # =========================================================
            # 3. Left pad to batch max length (generation-style)
            # =========================================================
            max_fused_L = max(s.size(1) for s in fused_samples_list)
            prompt_embeds = torch.zeros(B, max_fused_L, self.d_llm, device=device, dtype=self.model.dtype)
            prompt_attn_mask = torch.zeros(B, max_fused_L, device=device, dtype=torch.long)

            diffs = []
            for b in range(B):
                curr_L = fused_samples_list[b].size(1)
                diff = max_fused_L - curr_L
                diffs.append(diff)
                prompt_embeds[b, diff:] = fused_samples_list[b]
                prompt_attn_mask[b, diff:] = 1

            # =========================================================
            # 3b. Bio thinker: one-pass hidden thoughts for bio-latent tokens
            # =========================================================
            if use_biothinker and any(bio_latent_positions_list):
                # IMPORTANT: BioThinker is bidirectional; mask out the trailing <start_latent> token so it can't
                # (even trivially) influence bio-latent embeddings.
                bio_thinker_mask = prompt_attn_mask.clone()
                for b in range(B):
                    curr_L = int(fused_samples_list[b].size(1))
                    if curr_L <= 0:
                        continue
                    if (not use_coconut) and use_taskthinker:
                        # When TaskThinker is enabled (non-coconut), each sample ends with an appended <start_latent>.
                        start_latent_pos = int(diffs[b] + curr_L - 1)
                        if 0 <= start_latent_pos < bio_thinker_mask.size(1):
                            bio_thinker_mask[b, start_latent_pos] = 0

                thinker_out = self.bio_thinker(prompt_embeds, attention_mask=bio_thinker_mask)
                prompt_embeds_updated = prompt_embeds.clone()
                for b in range(B):
                    positions = bio_latent_positions_list[b]
                    if positions:
                        shifted = [p + diffs[b] for p in positions]
                        prompt_embeds_updated[b, shifted] = thinker_out[b, shifted].to(dtype=prompt_embeds_updated.dtype)
                if self.is_biothinker_gating and self.bio_thinker_gate is not None:
                    active_indices = [b for b in range(B) if bio_latent_block_spans_list[b] is not None]
                    if active_indices:
                        anchor_states = []
                        gate_inputs = []
                        spans_shifted = []
                        for b in active_indices:
                            span = bio_latent_block_spans_list[b]
                            anchor_pos = bio_latent_anchor_pos_list[b]
                            if span is None or anchor_pos is None:
                                continue
                            diff = int(diffs[b])
                            start, end = span
                            start_s = diff + int(start)
                            end_s = diff + int(end)
                            anchor_s = diff + int(anchor_pos)
                            anchor_states.append(prompt_embeds_updated[b, anchor_s])
                            gate_inputs.append(thinker_out[b, anchor_s])
                            spans_shifted.append((b, start_s, end_s))

                        if anchor_states:
                            anchor_states_t = torch.stack(anchor_states, dim=0)
                            gate_inputs_t = torch.stack(gate_inputs, dim=0)
                            gates = self.bio_thinker_gate(gate_inputs_t, out_dtype=prompt_embeds_updated.dtype)
                            for i, (b, start_s, end_s) in enumerate(spans_shifted):
                                if end_s <= start_s:
                                    continue
                                g = gates[i, 0, 0]
                                anchor = anchor_states_t[i].to(dtype=prompt_embeds_updated.dtype)
                                bio_block = prompt_embeds_updated[b, start_s:end_s].clone()
                                prompt_embeds_updated[b, start_s:end_s] = bio_block * g + anchor.unsqueeze(0) * (1.0 - g)
                prompt_embeds = prompt_embeds_updated

            # =========================================================
            # 4. Coconut latent-feedback refinement (optional based on presence of <latent>)
            # =========================================================
            has_latent = (input_ids == self.latent_id).any().item()
            if use_coconut and has_latent and attention_mask is not None:
                latent_positions = []
                final_bio_positions = []
                for b in range(B):
                    mol_len_b = mol_counts[b] * (self.num_queries + 2)
                    t_mask_b = attention_mask[b].bool()
                    rel_latent_indices = (input_ids[b][t_mask_b] == self.latent_id).nonzero(as_tuple=True)[0]
                    diff = diffs[b]
                    latent_positions.append((rel_latent_indices + mol_len_b + diff).tolist())
                    final_bio_positions.append([idx + diff for idx in bio_positions_list[b]])

                prompt_embeds = self._apply_latent_feedback(
                    prompt_embeds,
                    prompt_attn_mask,
                    latent_positions,
                    bio_positions=final_bio_positions,
                    refine_bio_tokens=refine_bio_tokens,
                )

            # =========================================================
            # 5. Task latent generation (when TaskThinker is enabled)
            # Each step: decode next token; if <end_latent> then append and stop,
            # otherwise append a new latent embedding (from hidden state) refined by TaskThinker.
            # =========================================================
            # NOTE: Coconut mode already uses <start_latent>/<latent>/<end_latent> in `input_ids`; avoid duplicating.
            if use_taskthinker and (not use_coconut):
                llm = self._get_actual_llm()
                backbone = llm.model
                # has_lora_in_backbone = any(hasattr(m, "lora_A") and hasattr(m, "lora_B") for m in backbone.modules())
                # if not has_lora_in_backbone:
                #     raise RuntimeError("Expected LoRA layers in `llm.model` (backbone), but none was detected.")
                lm_head = llm.lm_head

                new_samples = []
                latent_mask_starts = []
                latent_mask_ends = []
                task_latent_counts = []
                for b in range(B):
                    diff = diffs[b]
                    seq = prompt_embeds[b, diff:]  # [L_b, d]
                    if seq.size(0) == 0:
                        new_samples.append(seq)
                        task_latent_counts.append(0)
                        continue
                    # We appended <start_latent> at the end.
                    base_prefix = seq[:-1].unsqueeze(0).clone()  # [1, L-1, d]
                    latent_block = seq[-1:].unsqueeze(0).clone()  # [1, 1, d] (starts with <start_latent>)
                    latent_state_hist = []
                    bio_positions = bio_positions_list[b]
                    bioupdater_gate_cache = None

                    ended = False
                    for current_latent_step in range(int(self.task_latent_max_steps)):
                        full_seq = torch.cat([base_prefix, latent_block], dim=1)
                        full_mask = torch.ones(1, full_seq.size(1), device=device, dtype=torch.long)
                        # if self.mask_task_latent_steps > 0 and current_latent_step > 0:
                        #     mask_steps = min(self.mask_task_latent_steps, current_latent_step)
                        #     if self.mask_task_latent_implementation == 'mask':
                        #         full_mask[:, base_prefix.size(1) + 1 : base_prefix.size(1) + 1 + mask_steps] = 0
                        #     else:
                        #         raise RuntimeError(f"Unimplemented mask_task_latent_implementation: {self.mask_task_latent_implementation}")
                        # Task-latent token *sampling* does not need gradients; gradients are provided by the later GRPO
                        # log-prob forward on the full (prompt + completion) sequence.
                        # step_inputs = {"inputs_embeds": full_seq, "attention_mask": full_mask, "use_cache": False}
                        # self.total_inference_flops += FlopCountAnalysis(backbone, (), step_inputs).total()
                        
                        before_backbone = flop_counter.get_total_flops()
                        with torch.no_grad():
                            out = backbone(
                                inputs_embeds=full_seq,
                                attention_mask=full_mask,
                                return_dict=True,
                                use_cache=False,
                            )
                            last_hidden = out.last_hidden_state  # (1, L, d)
                            logits_last = lm_head(last_hidden[:, -1, :])  # (1, vocab)
                        
                        after_backbone = flop_counter.get_total_flops()
                        # just leave the all steps. tired.
                        self.sample_inner_flops.append(after_backbone - before_backbone)

                        next_id = int(logits_last.argmax(dim=-1).item())
                        if next_id == int(self.end_latent_id):
                            # print(current_latent_step)
                            latent_block = torch.cat([latent_block, end_latent_emb], dim=1)
                            ended = True
                            break

                        latent_state = last_hidden[:, -1:, :].to(dtype=torch.float32)
                        # corrupt here
                        # we corrupt every latent, including <start_latent>'s output
                        if self.mask_task_latent_steps > current_latent_step:
                            if self.mask_task_latent_implementation == 'noise':
                                ids = input_ids[b].to(dtype=torch.int64)
                                seed_val = int(((ids + 1) * 1315423911).sum().item()) & 0xFFFFFFFFFFFFFFFF
                                gen = torch.Generator(device=latent_state.device)
                                gen.manual_seed(seed_val)
                                noise = torch.randn(
                                    latent_state.shape,
                                    generator=gen,
                                    device=latent_block.device,
                                    dtype=torch.float32,
                                ) * float(self.mask_task_latent_noise_std)
                                latent_state = noise.to(dtype=latent_state.dtype)
                            elif self.mask_task_latent_implementation == 'zero':
                                latent_state = torch.zeros_like(latent_state)
                            else:
                                raise NotImplementedError(
                                    f"mask_task_latent_implementation={self.mask_task_latent_implementation} is not implemented. If you don't want to mask task latents, set `mask_task_latent_steps=0`.")
                        latent_state_hist.append(latent_state)
                        if refine_bio_tokens and bio_positions:
                            batched_bio = base_prefix[:, bio_positions].to(dtype=self.model.dtype)
                            batched_lat = torch.cat(latent_state_hist, dim=1).to(dtype=self.model.dtype)
                            if self.is_bioupdater_gating:
                                if bioupdater_gate_cache is None:
                                    if self.bio_updater_gate is None:
                                        raise RuntimeError(
                                            "is_bioupdater_gating=True but `bio_updater_gate` is not initialized."
                                        )
                                    bioupdater_gate_cache = self.bio_updater_gate(
                                        batched_lat[:, -1, :], out_dtype=batched_bio.dtype
                                    )
                                refined = self.bio_updater(batched_bio, batched_lat)
                                refined = refined * bioupdater_gate_cache + batched_bio * (1.0 - bioupdater_gate_cache)
                            else:
                                refined = self.bio_updater(batched_bio, batched_lat)
                            base_prefix[:, bio_positions] = refined.to(dtype=base_prefix.dtype)

                        new_latent = latent_state.to(dtype=self.model.dtype)
                        new_latent = self._taskthinker_with_gating(new_latent)
                        latent_block = torch.cat([latent_block, new_latent], dim=1)

                    if not ended:
                        latent_block = torch.cat([latent_block, end_latent_emb], dim=1)

                    n_task_latents = max(int(latent_block.size(1)) - 2, 0)
                    task_latent_counts.append(n_task_latents)

                    # if self.mask_task_latent_steps > 0 and latent_block.size(1) > 2:
                    #     mask_steps = min(self.mask_task_latent_steps, latent_block.size(1) - 2)
                    #     # print(mask_steps, self.mask_task_latent_steps, latent_block.size(1) - 2)
                    #     latent_block = latent_block.clone()
                    #     if self.mask_task_latent_implementation == 'noise':
                    #         ids = input_ids[b].to(dtype=torch.int64)
                    #         seed_val = int(((ids + 1) * 1315423911).sum().item()) & 0xFFFFFFFFFFFFFFFF
                    #         gen = torch.Generator(device=latent_block.device)
                    #         gen.manual_seed(seed_val)
                    #         noise = torch.randn(
                    #             latent_block[:, 1:1 + mask_steps, :].shape,
                    #             generator=gen,
                    #             device=latent_block.device,
                    #             dtype=torch.float32,
                    #         ) * float(self.mask_task_latent_noise_std)
                    #         latent_block[:, 1:1 + mask_steps, :] = noise.to(dtype=latent_block.dtype)
                    #     elif self.mask_task_latent_implementation == 'zero':
                    #         latent_block[:, 1:1 + mask_steps, :] = torch.zeros_like(latent_block[:, 1:1 + mask_steps, :])
                    #     elif self.mask_task_latent_implementation == 'mask':
                    #         # print(base_prefix.size(1))
                    #         latent_mask_starts.append(base_prefix.size(1) + 1)
                    #         latent_mask_ends.append(base_prefix.size(1) + 1 + mask_steps)
                    #     else:
                    #         raise NotImplementedError(
                    #             f"mask_task_latent_implementation={self.mask_task_latent_implementation} is not implemented. If you don't want to mask task latents, set `mask_task_latent_steps=0`.")
                            
                    if self.shuffle_task_latents and latent_block.size(1) > 2:
                        latent_block = latent_block.clone()
                        ids = input_ids[b].to(dtype=torch.int64)
                        seed_val = int(((ids + 1) * 1315423911).sum().item()) & 0xFFFFFFFFFFFFFFFF
                        gen = torch.Generator(device='cpu')
                        gen.manual_seed(seed_val)
                        perm = torch.randperm(latent_block.size(1) - 2, generator=gen).to(latent_block.device) + 1
                        latent_block[:, 1:-1, :] = latent_block[:, perm, :]
                        # print("shuffled")
                    
                    if corrupt_flags[b] and latent_block.size(1) > 2:
                        latent_block = latent_block.clone()
                        if float(corrupt_task_latent_noise_std) > 0.0:
                            ids = input_ids[b].to(dtype=torch.int64)
                            seed_val = int(((ids + 1) * 1315423911).sum().item()) & 0xFFFFFFFFFFFFFFFF
                            gen = torch.Generator(device=latent_block.device)
                            gen.manual_seed(seed_val)
                            noise = torch.randn(
                                latent_block[:, 1:-1, :].shape,
                                generator=gen,
                                device=latent_block.device,
                                dtype=torch.float32,
                            ) * float(corrupt_task_latent_noise_std)
                            latent_block[:, 1:-1, :] = noise.to(dtype=latent_block.dtype)
                        else:
                            latent_block[:, 1:-1, :] = torch.zeros_like(latent_block[:, 1:-1, :])

                    new_samples.append(torch.cat([base_prefix, latent_block], dim=1).squeeze(0))

                max_L = max(s.size(0) for s in new_samples) if new_samples else 0
                new_prompt_embeds = torch.zeros(B, max_L, self.d_llm, device=device, dtype=self.model.dtype)
                new_prompt_attn_mask = torch.zeros(B, max_L, device=device, dtype=torch.long)
                for b in range(B):
                    curr_L = new_samples[b].size(0)
                    diff = max_L - curr_L
                    new_prompt_embeds[b, diff:] = new_samples[b]
                    new_prompt_attn_mask[b, diff:] = 1
                    # print(curr_L)
                    # if latent_mask_starts and latent_mask_ends:
                    #     mask_start = latent_mask_starts[b] + diff
                    #     mask_end = latent_mask_ends[b] + diff
                    #     new_prompt_attn_mask[b, mask_start:mask_end] = 0
                    #     # print(mask_start, mask_end)

                prompt_embeds, prompt_attn_mask = new_prompt_embeds, new_prompt_attn_mask
                self._last_task_latent_counts = task_latent_counts
            elif refine_bio_tokens and (not (use_coconut and has_latent and attention_mask is not None)):
                # Special case: BioUpdater enabled but TaskThinker disabled -> do a single memory update once using the
                # last prompt-side token, without generating task-latents.
                active_indices = [b for b in range(B) if bio_positions_list[b] and fused_samples_list[b].size(1) > 0]
                if active_indices:
                    llm = self._get_actual_llm()
                    backbone = llm.model

                    embeds_in = prompt_embeds
                    out = backbone(
                        inputs_embeds=embeds_in,
                        attention_mask=prompt_attn_mask,
                        return_dict=True,
                        use_cache=False,
                    )
                    hidden_states = out.last_hidden_state  # (B, L, d)

                    bios = []
                    lats = []
                    for b in active_indices:
                        diff = int(diffs[b])
                        curr_L = int(fused_samples_list[b].size(1))
                        anchor_pos = diff + curr_L - 1
                        bios.append(embeds_in[b, [idx + diff for idx in bio_positions_list[b]]])
                        lats.append(hidden_states[b, anchor_pos].unsqueeze(0))

                    batched_bio = torch.nn.utils.rnn.pad_sequence(bios, batch_first=True)
                    batched_lat = torch.stack(lats, dim=0).to(dtype=self.model.dtype)
                    refined = self._bioupdater_with_gating(batched_bio.to(self.model.dtype), batched_lat)

                    embeds_out = embeds_in.clone()
                    for i, b in enumerate(active_indices):
                        diff = int(diffs[b])
                        positions = [idx + diff for idx in bio_positions_list[b]]
                        embeds_out[b, positions] = refined[i, : len(positions)].to(dtype=embeds_out.dtype)
                    prompt_embeds = embeds_out
            self.total_inference_flops = flop_counter.get_total_flops()
        return prompt_embeds, prompt_attn_mask

    @torch.no_grad()
    def generate(
        self,
        smiles_list: List[List[str]],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        同步更新后的生成方法：
        1. 支持变长分子和批量投影。
        2. 推理阶段自动改用左补齐 (Left Padding) 以确保生成质量。
        """
        # Backward compatible: allow passing List[str] (one SMILES per sample)
        if smiles_list and isinstance(smiles_list[0], str):
            smiles_list = [[s] for s in smiles_list]  # type: ignore[list-item]
        
        start_event = torch.cuda.Event(enable_timing=True)
        mid_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        # Reuse the shared embedding builder (and latent feedback) so vLLM can share the same code path.
        prompt_embeds, prompt_attn_mask = self.get_prompt_embeddings(
            smiles_list=smiles_list,
            input_ids=input_ids,
            attention_mask=attention_mask,
            refine_bio_tokens=kwargs.get("refine_bio_tokens", True),
        )

        mid_event.record()

        # prefill_inputs = {"inputs_embeds": prompt_embeds, "attention_mask": prompt_attn_mask, "use_cache": True}
        # prefill_flops = FlopCountAnalysis(self.model, (), prefill_inputs).total()

        with FlopCounterMode(display=False) as gen_counter:
                outputs = self.model.generate(
                    inputs_embeds=prompt_embeds,
                    attention_mask=prompt_attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=use_cache,
                    **kwargs
                )
                self.total_inference_flops += gen_counter.get_total_flops()

        end_event.record()
        # 必须同步，否则 CPU 拿不到准确的时间戳
        torch.cuda.synchronize()

        total_latency = start_event.elapsed_time(end_event)
        thinking_latency = start_event.elapsed_time(mid_event)
        generation_latency = mid_event.elapsed_time(end_event)

        self.total_time.append(total_latency)
        self.latent_time.append(thinking_latency)
        self.text_time.append(generation_latency)

        gen_len = outputs.shape[1]
        # self.total_inference_flops += prefill_step_flops + (gen_len * decode_single_step_flops)
        self.sample_step_latent_flops.append(gen_len)

        self.sample_latent_flops.append(self.total_inference_flops)

        return outputs


if __name__ == "__main__":
    # Test Initialization
    mol_config = {
        'num_queries': 8,
        'input_dim': 768,
        'num_heads': 2
    }
    model = Qwen3MoleculeLLM(
        qwen_model_name="/zengdaojian/zhangjia/BioLatent/Qwen4B",
        mol_config=mol_config
    ).cuda()
    print("Stage 3 Model Initialized Successfully!")
