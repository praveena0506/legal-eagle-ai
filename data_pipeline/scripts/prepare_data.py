import sys
import os

# 👇 FIX: Add project root to path
sys.path.append(os.getcwd())

import torch
import re
from pathlib import Path
from collections import Counter
# Now this import will work:
from data_pipeline.scripts.db_utils import fetch_all_cases

# ... (Keep the rest of the file exactly the same) ...

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_FILE = BASE_DIR / "processed_data.pt"


def clean_text(text):
    """
    Simple text cleaning: Lowercase and remove special characters.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


class SimpleTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {"<PAD>": 0, "<UNK>": 1}

    def build_vocab(self, texts):
        print("📊 Building Vocabulary...")
        all_words = []
        for text in texts:
            cleaned = clean_text(text)
            tokens = cleaned.split()
            all_words.extend(tokens)

        most_common = Counter(all_words).most_common(self.vocab_size - 2)

        for idx, (word, _) in enumerate(most_common):
            self.vocab[word] = idx + 2

        print(f"✅ Vocab built with {len(self.vocab)} words.")

    def encode(self, text, max_len=512):
        cleaned = clean_text(text)
        tokens = cleaned.split()
        token_ids = [self.vocab.get(t, 1) for t in tokens[:max_len]]

        if len(token_ids) < max_len:
            token_ids += [0] * (max_len - len(token_ids))

        return token_ids


def run_data_preparation():
    print("⚙️ Starting Data Preparation (Source: MongoDB Cloud)...")

    # 1. Fetch from Cloud ☁️
    raw_data = fetch_all_cases()
    print(f"📥 Downloaded {len(raw_data)} total cases from MongoDB.")

    texts = []
    labels = []

    # 2. Filter only Labeled Data
    for record in raw_data:
        verdict = record.get("verdict", "Unknown")
        text = record.get("text", "")

        if verdict == "Allowed":
            texts.append(text)
            labels.append(1)  # Win
        elif verdict == "Dismissed":
            texts.append(text)
            labels.append(0)  # Lose

    print(f"✅ Found {len(texts)} usable (labeled) cases.")

    if not texts:
        print("⚠️ No labeled data found yet! Run the Scraper first.")
        return

    # 3. Tokenize
    tokenizer = SimpleTokenizer(vocab_size=2000)
    tokenizer.build_vocab(texts)

    input_ids = []
    for text in texts:
        input_ids.append(tokenizer.encode(text))

    # 4. Save Processed Tensors
    data_bundle = {
        "vocab": tokenizer.vocab,
        "data": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

    torch.save(data_bundle, PROCESSED_FILE)
    print(f"💾 Saved processed data to {PROCESSED_FILE}")


if __name__ == "__main__":
    run_data_preparation()