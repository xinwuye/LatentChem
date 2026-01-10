"""
Embed only the "protein sequence" files using ESM-2 650M:
  - protein_function.*
  - catalytic_activity.*
  - domain_motif.*

Saves per-input HDF5: <basename>_esm2_650m.h5
"""
import os, sys, json, re
from glob import glob
from tqdm import tqdm
import h5py

import torch
import esm


INPUT_DIR = "Mol-Instructions/data/proteins/Protein-oriented_Instructions"   # folder with json files
OUTPUT_DIR = "embeddings_proteins"   # where to write .h5 outputs
FILES_TO_PROCESS = [
    "protein_function",
    "catalytic_activity",
    "domain_motif"
]
MODEL_NAME = "esm2_t33_650M_UR50D"
REP_LAYER = 33
POOL_METHOD = "mean"        # "mean" | "bos" | "per_token"
BATCH_SIZE = 8              
USE_FP16 = True             # use float16 on GPU if available
MIN_SEQ_LEN = 20
MAX_SEQ_LEN = 2500
AA_RE = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$', re.IGNORECASE)


def list_target_files(input_dir):
    # Helper function for mentioning the files which will be used in creating the embeddings
    candidates = []
    for base in FILES_TO_PROCESS:
        patterns = [os.path.join(input_dir, base + ext) for ext in (".json",".jsonl",".ndjson",".json.gz",".jsonl.gz")]
        for p in patterns:
            candidates.extend(glob(p))
    return sorted(set(candidates))

def load_json_or_jsonl(path):
    # Load the input files
    items = []
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    if path.endswith(".jsonl") or path.endswith(".ndjson") or path.endswith(".jsonl.gz") or path.endswith(".ndjson.gz"):
        with opener(path, "rt", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                items.append(json.loads(ln))
    else:
        with opener(path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # Try common keys
                for k in ("data","examples","items"):
                    if k in data and isinstance(data[k], list):
                        items = data[k]; break
                if not items:
                    # If no common keys: assume dict-of-lists -> combine
                    for v in data.values():
                        if isinstance(v, list):
                            items.extend(v)
                    if not items:
                        items = [data]
            else:
                items = [data]
    return items

def extract_sequence_from_example(ex):
    # Look for canonical fields first
    for key in ("input","protein_sequence","sequence"):
        v = ex.get(key) if isinstance(ex, dict) else None
        if isinstance(v, str):
            s = v.replace(" ", "").replace("\n", "").upper()
            if MIN_SEQ_LEN <= len(s) <= MAX_SEQ_LEN and AA_RE.match(s):
                return s
            # If the input contains a leading label then sequence - try regex
            m = re.search(r"([ACDEFGHIKLMNPQRSTVWY]{%d,})" % MIN_SEQ_LEN, v, re.IGNORECASE)
            if m:
                s2 = m.group(1).replace("\n","").upper()
                if MIN_SEQ_LEN <= len(s2) <= MAX_SEQ_LEN:
                    return s2
    # Look for AA-like substrings in all strings
    if isinstance(ex, dict):
        for v in ex.values():
            if isinstance(v, str):
                m = re.search(r"([ACDEFGHIKLMNPQRSTVWY]{%d,})" % MIN_SEQ_LEN, v, re.IGNORECASE)
                if m:
                    s = m.group(1).replace("\n","").upper()
                    if MIN_SEQ_LEN <= len(s) <= MAX_SEQ_LEN:
                        return s
    return None


def load_model(device):
    # Load the ESM-2 650M model and set it to evaluation
    print("Loading model", MODEL_NAME, "to", device)
    model, alphabet = esm.pretrained.__dict__[MODEL_NAME]()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if device == "cuda":
        model = model.to(device)
        if USE_FP16:
            model.half()
    return model, batch_converter


def embed_file(path, model, batch_converter, device):
    # Embed each of the input files into a separate file
    basename = os.path.basename(path)
    stem = os.path.splitext(basename)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{stem}_esm2_650m.h5")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n=== Processing {basename} ===")
    items = load_json_or_jsonl(path)
    print("Loaded items:", len(items))

    seq_pairs = []
    for idx, ex in enumerate(items):
        seq = extract_sequence_from_example(ex)
        if seq:
            seq_pairs.append((idx, seq))
    print("Extracted sequences:", len(seq_pairs))
    if len(seq_pairs) == 0:
        print("No valid sequences found; skipping.")
        return

    # Create the HDF5 file for saving the embeddings
    h5f = h5py.File(out_path, "w")
    meta = h5f.create_group("meta")
    emb = h5f.create_group("embeddings")

    device_t = torch.device(device)
    for i in tqdm(range(0, len(seq_pairs), BATCH_SIZE), desc=basename):
        batch = seq_pairs[i:i+BATCH_SIZE]
        names_seqs = [(f"ex{p[0]}", p[1]) for p in batch]
        labels, seqs, toks = batch_converter(names_seqs)
        toks = toks.to(device_t)
        with torch.no_grad():
            results = model(toks, repr_layers=[REP_LAYER], return_contacts=False)
            reprs = results["representations"][REP_LAYER]
            for j, (label, seq_text) in enumerate(zip(labels, seqs)):
                token_repr = reprs[j]           # (L, D)
                seq_len = len(seq_text)
                residue_repr = token_repr[1: seq_len+1].cpu().numpy()
                name = label
                if POOL_METHOD == "per_token":
                    emb.create_dataset(name, data=residue_repr, compression="gzip")
                elif POOL_METHOD == "mean":
                    emb.create_dataset(name, data=residue_repr.mean(axis=0), compression="gzip")
                elif POOL_METHOD == "bos":
                    emb.create_dataset(name, data=token_repr[0].cpu().numpy(), compression="gzip")
                meta.create_dataset(f"{name}_meta", data=json.dumps({
                    "example_idx": int(label.replace("ex","")),
                    "seq_len": seq_len,
                    "source_file": basename
                }).encode("utf8"))
    h5f.close()
    print("Wrote:", out_path) # Save embedding file

def main():
    # Set the device; default: CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print("Device:", device)

    # Load the model
    model, batch_converter = load_model(device) 

    # Mention target files
    files = list_target_files(INPUT_DIR) 
    if not files:
        print("No target files found in", INPUT_DIR)
        sys.exit(1)
    print("Target files:", files)

    # Create embeddings for each target file, i.e. the ones containing protein sequences. 
    for f in files:
        embed_file(f, model, batch_converter, device)

if __name__ == "__main__":
    main()