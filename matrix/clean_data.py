import os
import re

logp_dir = "/zengdaojian/zhangjia/BioLatent/Bio-LatentCOT/refined/solubility"

keep_pattern = re.compile(r"\d+_refined_9\.npy$")
delete_pattern = re.compile(r"\d+_refined_\d+\.npy$")

for fname in os.listdir(logp_dir):
    if delete_pattern.match(fname) and not keep_pattern.match(fname):
        path = os.path.join(logp_dir, fname)
        os.remove(path)
        print("Deleted:", fname)

