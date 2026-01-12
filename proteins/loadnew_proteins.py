# smi_ted_light/loadnew.py
"""
Robust loader for the SMI-TED encoder and a minimal Protein encoder.

Exposes:
- load_smi_ted(folder, ckpt_filename, vocab_filename, encoder_type="smiles")
- load_protein_ted(output_dim=768, max_len=2048, freeze=False)
Both return an object with a .encode(smiles_list) method returning a nested
list of tensors like: [[Tensor(L1, D), Tensor(L2, D)], [ ... ], ...]
(which is what model_stage3 expects).
"""

import os
import torch
import torch.nn as nn
import warnings

# Try importing original Smi_ted if available in package; fallback to a shim.
try:
    # If your original smi-ted package exists, import its loader/class
    # The user's previous code had `load_smi_ted` in a top-level module; try to import if present.
    from .smi_ted import Smi_ted, load_smi_ted as _orig_load_smi_ted  # pragma: no cover
    _HAVE_ORIG = True
except Exception:
    _HAVE_ORIG = False


class ProteinEncoder(nn.Module):
    """
    Minimal protein encoder that turns protein amino-acid sequences into per-residue embeddings.
    Provides a compatible `.encode(list_of_list_of_proteins)` API:
      Input: smiles_list = [[prot1, prot2], [prot1], ...]
      Output: nested list: for each sample -> list of Tensors with shape [L, output_dim]
    Use this as a simple trainable encoder. If you have a pretrained encoder, replace this loader.
    """
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # canonical 20 AA
    AA2IDX = {a: i + 1 for i, a in enumerate(AA_LIST)}  # reserve 0 for padding

    def __init__(self, output_dim: int = 768, max_len: int = 2048, padding_idx: int = 0):
        super().__init__()
        self.output_dim = int(output_dim)
        self.max_len = int(max_len)
        self.vocab_size = len(self.AA2IDX) + 1
        self.padding_idx = padding_idx

        # small per-AA embedding + optional proj
        self.embed = nn.Embedding(self.vocab_size, min(256, self.output_dim), padding_idx=self.padding_idx)
        self.proj = nn.Linear(self.embed.embedding_dim, self.output_dim)

        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, idx_tensor):
        # idx_tensor: LongTensor (batch, L)
        x = self.embed(idx_tensor)  # (B, L, E)
        x = self.proj(x)  # (B, L, D)
        return x

    def _tokenize_sequence(self, seq: str):
        """Map sequence string -> list of int token ids (no special tokens)."""
        seq = seq.strip().upper()
        ids = []
        for ch in seq:
            ids.append(self.AA2IDX.get(ch, 0))
        return ids

    def encode(self, nested_prots):
        """
        Accepts nested list-of-list: [[prot1, prot2], [prot1], ...]
        Returns nested list: [[Tensor(L1, D), Tensor(L2, D)], ...]
        All tensors are on CPU (caller can .to(device) afterwards).
        """
        outputs = []
        device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device("cpu")

        for sample in nested_prots:
            sample_out = []
            for prot in sample:
                if not prot or not isinstance(prot, str):
                    sample_out.append(torch.zeros(0, self.output_dim, device=device))
                    continue
                ids = self._tokenize_sequence(prot)
                if len(ids) == 0:
                    sample_out.append(torch.zeros(0, self.output_dim, device=device))
                    continue
                with torch.no_grad():
                    idx_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
                    emb = self.forward(idx_tensor).squeeze(0)  # (L, D)
                    sample_out.append(emb.cpu())
            outputs.append(sample_out)
        return outputs


def load_protein_ted(output_dim: int = 768, max_len: int = 2048, freeze: bool = True):
    """
    Build or load a lightweight protein encoder with `.encode(...)` API.
    Returns the encoder module (in eval mode if freeze=True).
    """
    encoder = ProteinEncoder(output_dim=output_dim, max_len=max_len)
    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
    return encoder


def load_smi_ted(folder="./smi_ted_light",
                  ckpt_filename="smi-ted-Light_40.pt",
                  vocab_filename="bert_vocab_curated.txt",
                  encoder_type: str = "smiles",
                  protein_output_dim: int = 768,
                  protein_max_len: int = 2048,
                  freeze_protein: bool = True):
    """
    Unified loader:
      - encoder_type == "smiles": attempt to load your original smi-ted checkpoint (if available)
      - encoder_type == "protein": return a ProteinEncoder (lightweight)
    The returned object must implement `.encode(list_of_list)` returning nested lists of tensors
    where each tensor has shape (L, D) and D matches mol_config['input_dim'] / projector.input_dim.
    """
    encoder_type = (encoder_type or "smiles").lower()
    if encoder_type not in ("smiles", "protein"):
        raise ValueError("encoder_type must be 'smiles' or 'protein'")

    if encoder_type == "protein":
        # Create a small protein encoder
        enc = load_protein_ted(output_dim=protein_output_dim, max_len=protein_max_len, freeze=freeze_protein)
        # For compatibility with older code that expects attributes:
        enc.output_dim = protein_output_dim
        enc.max_len = protein_max_len
        return enc

    # Otherwise attempt to load original SMI-TED model (if available)
    if _HAVE_ORIG:
        try:
            # Try to use original loader if present
            return _orig_load_smi_ted(folder=folder, ckpt_filename=ckpt_filename, vocab_filename=vocab_filename)
        except Exception as e:
            warnings.warn(f"Original smi-ted loader failed: {e}. Falling back to ProteinEncoder shim.")
            # fall through to default shim as fallback

    # Fallback: if original Smi_ted module is not available, create a dummy encoder
    warnings.warn("Original SMI-TED not available; returning a ProteinEncoder-style dummy (use encoder_type='protein' explicitly to avoid this).")
    dummy = load_protein_ted(output_dim=protein_output_dim, max_len=protein_max_len, freeze=True)
    dummy.output_dim = protein_output_dim
    return dummy