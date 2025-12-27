import os
import requests
import tarfile
from tqdm import tqdm  # Progress bar

# 1. Define the target (This is where Chroma is trying to save it)
# We use the path we saw in your error log: C:\Users\prave\.cache\chroma...
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "chroma", "onnx_models", "all-MiniLM-L6-v2")
FILE_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
DEST_FILE = os.path.join(CACHE_DIR, "onnx.tar.gz")


def force_download_model():
    # Create the folder if it doesn't exist
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"Created directory: {CACHE_DIR}")

    print(f"🚀 Starting download to: {DEST_FILE}")
    print("This may take time, but it won't timeout!")

    # Download with NO timeout
    response = requests.get(FILE_URL, stream=True, timeout=None)

    # Save the file
    total_size = int(response.headers.get('content-length', 0))
    with open(DEST_FILE, "wb") as file, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    print("\n✅ Download Complete. Extracting...")

    # Extract the tar.gz file
    with tarfile.open(DEST_FILE, "r:gz") as tar:
        tar.extractall(path=CACHE_DIR)

    print("🎉 Extraction Complete! The model is manually installed.")
    print("You can now run your ETL script.")


if __name__ == "__main__":
    force_download_model()