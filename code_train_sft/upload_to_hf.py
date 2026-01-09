from huggingface_hub import HfApi

api = HfApi()

# Upload all files in a folder
api.upload_folder(
    folder_path="/path", # change the path
    repo_id="blc-org/A-secret-model-repo",
    repo_type="model"  # Use "dataset" or "space" if uploading those instead
)