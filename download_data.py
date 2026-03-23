"""
Download the Higgs Boson Detection 2025 dataset from Kaggle.

Prerequisites:
    pip install kaggle
    Place your kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)

Usage:
    python download_data.py
"""

import os
import subprocess
import zipfile
from pathlib import Path


def main():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    competition = "higgs-boson-detection-2025"

    print(f"Downloading {competition} dataset to {data_dir}...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(data_dir)],
        check=True,
    )

    # Unzip any downloaded zip files
    for zip_file in data_dir.glob("*.zip"):
        print(f"Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(data_dir)
        zip_file.unlink()
        print(f"Removed {zip_file.name}")

    print("\nDataset files:")
    for f in sorted(data_dir.iterdir()):
        if f.name != ".gitkeep":
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.1f} MB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
