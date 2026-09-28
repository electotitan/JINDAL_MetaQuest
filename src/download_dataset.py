"""
download_dataset.py — Download Severstal Steel Defect Detection from Kaggle
=============================================================================
This script provides helper commands to download the real 13GB Severstal dataset
from Kaggle when you are ready to train on the full competition data.

Prerequisites:
  1. A Kaggle account (kaggle.com)
  2. Join the competition: https://www.kaggle.com/c/severstal-steel-defect-detection/rules
  3. Create an API token: Account Settings -> API -> "Create New Token" (downloads kaggle.json)
  4. Place kaggle.json in ~/.kaggle/ (Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json)

Run with:
    python src/download_dataset.py
"""

import os
import sys
import subprocess
import zipfile


def check_kaggle_token():
    user_home = os.path.expanduser("~")
    kaggle_path = os.path.join(user_home, ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_path):
        print("=" * 70)
        print("KAGGLE API TOKEN NOT FOUND")
        print("=" * 70)
        print(f"Expected location: {kaggle_path}\n")
        print("Steps to obtain:")
        print("  1. Log in to https://www.kaggle.com")
        print("  2. Go to: https://www.kaggle.com/settings -> Click 'Create New Token'")
        print(f"  3. Move the downloaded kaggle.json into: {os.path.dirname(kaggle_path)}")
        print("  4. Accept the rules: https://www.kaggle.com/c/severstal-steel-defect-detection/rules")
        print("=" * 70)
        return False
    return True


def download_and_extract():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    if not check_kaggle_token():
        print("\nNote: You can run 'python src/generate_sample_data.py' to work")
        print("with realistic sample data right now without Kaggle credentials!")
        return

    print("Downloading Severstal dataset via Kaggle CLI (~13 GB)...")
    cmd = [
        "kaggle", "competitions", "download",
        "-c", "severstal-steel-defect-detection",
        "-p", data_dir
    ]

    try:
        subprocess.run(cmd, check=True)
        zip_path = os.path.join(data_dir, "severstal-steel-defect-detection.zip")
        if os.path.exists(zip_path):
            print(f"Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("Extraction complete!")
            # Also extract train_images.zip if nested
            train_zip = os.path.join(data_dir, "train_images.zip")
            if os.path.exists(train_zip):
                print("Extracting train_images.zip...")
                with zipfile.ZipFile(train_zip, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(data_dir, "train_images"))
    except Exception as e:
        print(f"Error during download/extraction: {e}")


if __name__ == "__main__":
    download_and_extract()
