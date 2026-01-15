# ---------------- Protein HDF5 loader (paste after imports) ----------------
import glob
import h5py
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import os
import warnings

class ProteinH5Encoder(nn.Module):
    """
    Loads existing ESM embeddings saved as <basename>_esm2_650m.h5
    under group '/embeddings'. Exposes:
      - get(key) -> Tensor (L, D) or (1, D)
      - encode(list_of_list_keys) -> List[List[Tensor]] (samples -> per-protein tensors)
      - encode_batch_keys(keys) -> (padded_tensor, mask)
    Keeps no trainable parameters (so your existing freeze code is unaffected).
    """
    def __init__(self, folder: str, pool: str = "none", device: Optional[str] = None, lazy_open: bool = True):
        super().__init__()
        self.folder = folder
        self.pool = pool
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lazy_open = lazy_open

        paths = sorted(glob.glob(os.path.join(folder, "*_esm2_650m.h5")))
        if not paths:
            raise FileNotFoundError(f"No *_esm2_650m.h5 found in {folder!r}")

        self._index = {}
        self._handles = {}

        for fp in paths:
            try:
                with h5py.File(fp, "r") as fh:
                    emb = fh.get("embeddings")
                    if emb is None:
                        continue
                    base = os.path.basename(fp).replace("_esm2_650m.h5", "")
                    for ds in emb.keys():
                        self._index[ds] = (fp, ds)
                        self._index[f"{base}/{ds}"] = (fp, ds)
            except Exception as e:
                warnings.warn(f"While indexing {fp}: {e}")

        if not self._index:
            raise RuntimeError("No embedding datasets found under '/embeddings' in discovered .h5 files")

        if not self.lazy_open:
            for fp, _ in set(self._index.values()):
                self._handles[fp] = h5py.File(fp, "r")

        # try to infer embedding dim
        try:
            sample_fp, sample_ds = next(iter(self._index.values()))
            with h5py.File(sample_fp, "r") as fh:
                arr = fh["embeddings"][sample_ds][()]
            self.embedding_dim = int(arr.shape[-1])
        except Exception:
            self.embedding_dim = None

    def _open(self, fp: str):
        if fp in self._handles:
            return self._handles[fp]
        fh = h5py.File(fp, "r")
        if not self.lazy_open:
            self._handles[fp] = fh
        return fh

    def _resolve_array(self, key: str):
        # exact match
        if key in self._index:
            fp, ds = self._index[key]
            fh = self._open(fp)
            return fh["embeddings"][ds][()]

        # allow "basename/key" or "basename:key"
        if "/" in key or ":" in key:
            sep = "/" if "/" in key else ":"
            filepart, dsname = key.split(sep, 1)
            candidates = [fp for fp, _ in self._index.values() if os.path.basename(fp).startswith(filepart)]
            if candidates:
                fh = self._open(candidates[0])
                if "embeddings" in fh and dsname in fh["embeddings"]:
                    return fh["embeddings"][dsname][()]

        # fallback (slow)
        for fp, ds in self._index.values():
            if ds == key:
                fh = self._open(fp)
                return fh["embeddings"][ds][()]

        raise KeyError(f"Protein embedding key not found: {key}")

    def get(self, key: str) -> torch.Tensor:
        arr = self._resolve_array(key)
        t = torch.from_numpy(arr).to(dtype=torch.float32, device=self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() == 2:
            if self.pool == "mean":
                t = t.mean(dim=0, keepdim=True)
            elif self.pool == "bos":
                t = t[0:1]
            # pool == "none" -> keep (L, D)
        else:
            raise RuntimeError(f"Unsupported embedding rank {t.ndim} for key {key}")
        return t

    def encode(self, protein_list: List[List[str]]) -> List[List[torch.Tensor]]:
        out = []
        for sample in protein_list:
            if isinstance(sample, str):
                sample = [sample]
            sample_out = []
            for key in sample:
                try:
                    t = self.get(key)
                except KeyError:
                    alt = f"ex{key}" if key.isdigit() else key
                    try:
                        t = self.get(alt)
                    except KeyError:
                        raise KeyError(f"Couldn't resolve protein key '{key}'")
                sample_out.append(t)
            out.append(sample_out)
        return out

    def encode_batch_keys(self, keys: List[str], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        tensors = [self.get(k) for k in keys]
        Ls = [t.size(0) for t in tensors]
        Lmax = max(Ls)
        D = tensors[0].size(1)
        out = torch.full((len(tensors), Lmax, D), float(pad_value), device=self.device, dtype=torch.float32)
        mask = torch.zeros(len(tensors), Lmax, dtype=torch.bool, device=self.device)
        for i, t in enumerate(tensors):
            out[i, : t.size(0)] = t
            mask[i, : t.size(0)] = True
        return out, mask

    # no trainable params
    def parameters(self, recurse: bool = True):
        return iter([])

    def named_parameters(self, recurse: bool = True):
        return iter([])

    def eval(self):
        super().eval()
        return self

def load_protein_h5_encoder(folder: str, ckpt_filename: Optional[str] = None, device: Optional[str] = None, pool: str = "none"):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return ProteinH5Encoder(folder=folder, pool=pool, device=dev, lazy_open=True)
# ---------------- end loader ----------------