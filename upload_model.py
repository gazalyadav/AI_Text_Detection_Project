from huggingface_hub import HfApi

api = HfApi()

api.create_repo(
    repo_id="gazalyadav/ai-text-detector-deberta",
    repo_type="model",
    exist_ok=True
)

api.upload_folder(
    folder_path="src/models/saved/deberta",
    repo_id="gazalyadav/ai-text-detector-deberta",
    repo_type="model"
)

print("DeBERTa model uploaded successfully!")