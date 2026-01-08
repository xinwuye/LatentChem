from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from peft import PeftModel
from safetensors.torch import save_file
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--qwen_model_name", type=str, required=True)
parser.add_argument("--lora_weights_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

qwen_model_name = args.qwen_model_name
lora_weights_path = args.lora_weights_path
output_dir = args.output_dir

config = PretrainedConfig.from_pretrained(qwen_model_name)

tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)

extra_tokens = ["<mol_start>", "<mol_end>", "<latent>", "<start_latent>", "<end_latent>"]
tokenizer.add_tokens(extra_tokens)

model = AutoModelForCausalLM.from_pretrained(
    qwen_model_name,
    torch_dtype=torch.float32,
)

model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, lora_weights_path)
model = model.merge_and_unload()

embed_module = model.get_input_embeddings()
weight = embed_module.weight.detach().cpu()

embedding_save_path = os.path.join(output_dir, "embeddings.safetensors")
save_file({"weight": weight}, embedding_save_path)
print(f"Embedding weights saved to {embedding_save_path}")

tokenizer.save_pretrained(output_dir)
print(f"Tokenizer saved to {output_dir}")

config.save_pretrained(output_dir)
print(f"Config saved to {output_dir}")

model.save_pretrained(output_dir)