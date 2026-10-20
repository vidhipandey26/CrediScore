import argparse, os
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--folder", required=True)    # jaise runs/exp1
ap.add_argument("--repo_id", required=True)   # jaise your-username/truthlens-image
ap.add_argument("--public", action="store_true")
a = ap.parse_args()
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(a.repo_id, repo_type="model", private=not a.public, exist_ok=True)
api.upload_folder(folder_path=a.folder, repo_id=a.repo_id, repo_type="model")
print("uploaded ->", f"https://huggingface.co/{a.repo_id}")