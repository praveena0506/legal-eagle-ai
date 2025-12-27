import os
import shutil
from pathlib import Path
from datasets import load_dataset

# --- Configuration ---
# We take 200 cases to keep training fast on Colab
dataset_split = 'train[:200]'
base_path = Path("../raw_data")  # Goes up one level to data_pipeline/raw_data

# Define Class Folders
class_1_path = base_path / "appeal_accepted"  # Label 1
class_0_path = base_path / "appeal_rejected"  # Label 0


def setup_directories():
    # Clean up old data if it exists so we don't duplicate
    if base_path.exists():
        shutil.rmtree(base_path)

    class_1_path.mkdir(parents=True, exist_ok=True)
    class_0_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Created directories at: {base_path}")


def fetch_ildc_data():
    setup_directories()

    print("🌍 Connecting to Hugging Face Hub...")
    print(f"⏳ Downloading ILDC dataset ({dataset_split})...")

    # Load the Indian Legal Documents Corpus
    dataset = load_dataset("ildc", "ildc_single", split=dataset_split, trust_remote_code=True)

    print(f"✅ Downloaded {len(dataset)} cases. Saving to disk...")

    saved_count = 0

    for i, case in enumerate(dataset):
        text = case['text']
        label = case['label']  # 1 (Accepted) or 0 (Rejected)

        # Basic cleaning to save space
        text = " ".join(text.split())

        filename = f"case_{i}.txt"

        if label == 1:
            file_path = class_1_path / filename
        else:
            file_path = class_0_path / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        saved_count += 1
        if saved_count % 50 == 0:
            print(f"   ...saved {saved_count} files")

    print(f"\n🎉 Success! {saved_count} real legal cases are ready.")


if __name__ == "__main__":
    fetch_ildc_data()