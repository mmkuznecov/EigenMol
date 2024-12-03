import os
import subprocess
import yaml
import argparse


def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def configure_git_credentials(user, access_token):
    """Configure Git credentials for Hugging Face."""
    print("Configuring Git credentials...")
    git_credentials = f"https://{user}:{access_token}@huggingface.co"
    os.environ["GIT_ASKPASS"] = "echo"
    os.environ["GIT_USERNAME"] = user
    os.environ["GIT_PASSWORD"] = access_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = access_token
    os.environ["HUGGINGFACE_TOKEN"] = access_token
    return git_credentials


def clone_or_pull_repo(repo_url, local_path, credentials_url):
    """Clone the repository if it doesn't exist or pull the latest changes."""
    if not os.path.exists(local_path):
        print("Cloning the repository...")
        subprocess.run(["git", "lfs", "install"], check=True)
        # Modify the repo URL with credentials for authentication
        auth_url = repo_url.replace("https://huggingface.co", credentials_url)
        subprocess.run(["git", "clone", auth_url, local_path], check=True)
    else:
        print("Repository already cloned. Pulling the latest changes...")
        subprocess.run(["git", "-C", local_path, "pull"], check=True)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Fetch storages from Hugging Face repository."
    )
    parser.add_argument(
        "--config", required=True, help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    hf_repo = config["hf_repo"]
    local_path = config.get(
        "local_path", "storages"
    )  # Default local path to 'storages'
    user = config["user"]
    access_token = config["access_token"]

    # Configure Git credentials
    credentials_url = configure_git_credentials(user, access_token)

    # Clone or pull the repository
    clone_or_pull_repo(hf_repo, local_path, credentials_url)


if __name__ == "__main__":
    main()
