import os
from pathlib import Path

import yaml

with open(os.path.dirname(__file__) + "/../config/config.yaml", "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.FullLoader)


def fetch_healthy_dataset(url, output):
    """Download and unzip healthy data from Google Drive."""
    if not output.exists():
        output.mkdir(parents=True)
    # Download data
    os.system(f"gdown {url} -O data/raw/data_healthy.zip")
    # Check if data is downloaded
    assert os.path.isfile("data/raw/data_healthy.zip")
    # Unzip data
    os.system("unzip data/raw/data_healthy -d data/raw/")
    # Remove zip file and __MACOSX folder
    os.remove("data/raw/data_healthy.zip")
    os.system("rm -rf data/raw/__MACOSX")
    print("Data downloaded and unzipped successfully!")


def fetch_unhealthy_dataset(url, output):
    """Download and unzip unhealthy data from Google Drive."""
    if not output.exists():
        output.mkdir(parents=True)
    # Download data
    os.system(f"gdown {url} -O data/raw/data_unhealthy.zip")
    # Check if data is downloaded
    assert os.path.isfile("data/raw/data_unhealthy.zip")
    # Unzip data
    os.system("unzip data/raw/data_unhealthy -d data/raw/")
    # Remove zip file and __MACOSX folder
    os.remove("data/raw/data_unhealthy.zip")
    os.system("rm -rf data/raw/__MACOSX")
    print("Data downloaded and unzipped successfully!")


if __name__ == "__main__":
    output = Path("data/raw")

    # Fetch healthy data
    # fetch_healthy_dataset(config["datasets"]["url_data_normal"], output)
    # Fetch pathological data
    fetch_unhealthy_dataset(config["datasets"]["url_data_patho"], output)
