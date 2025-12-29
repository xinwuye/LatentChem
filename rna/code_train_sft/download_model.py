from huggingface_hub import snapshot_download

# export HF_ENDPOINT=https://hf-mirror.com

MY_TOKEN = "hf_XXX" # token 

print("Starting download... this might take a while for 16GB.")

snapshot_download(
    repo_id="Qwen/Qwen3-8B-Base",
    local_dir="models/Qwen3-8B-Base",
    token=MY_TOKEN,
    max_workers=8
)

print("Finished! Your model is in models/Qwen3-8B-Base")