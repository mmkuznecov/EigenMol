import os
import yaml
import argparse
from huggingface_hub import upload_file, HfApi
from tqdm import tqdm
import time


def upload_storages_to_hf(config_path):

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    username = config["username"]
    repo_name = config["repo_name"]
    local_folder = config["local_folder"]

    api = HfApi()

    repo_id = f"{username}/{repo_name}"
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    files_to_upload = [
        os.path.join(root, file)
        for root, _, files in os.walk(local_folder)
        for file in files
    ]

    print(f"Uploading {len(files_to_upload)} files to {repo_id}...")

    for file_path in tqdm(files_to_upload, desc="Uploading files", unit="file"):
        time.sleep(0.5)
        relative_path = os.path.relpath(file_path, local_folder)
        try:
            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"storages/{relative_path}",  # Preserve folder structure
                repo_id=repo_id,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"Failed to upload {file_path}: {e}")

    print(f"Successfully uploaded '{local_folder}' to {repo_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload storages to Hugging Face Hub.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    upload_storages_to_hf(args.config)
