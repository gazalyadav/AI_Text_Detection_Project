from huggingface_hub import HfApi

api = HfApi()

# First create the repo
api.create_repo(
    repo_id="gazalyadav/ai-text-detector-roberta",
    repo_type="model",
    exist_ok=True
)

# Then upload
api.upload_folder(
    folder_path="src/models/saved/roberta",
    repo_id="gazalyadav/ai-text-detector-roberta",
    repo_type="model"
)

print("Model uploaded successfully!")