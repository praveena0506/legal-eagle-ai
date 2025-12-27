import os
import torch
from pathlib import Path
import re

# --- Configuration ---
MAX_LEN = 512  # Maximum words per case (truncates longer ones)
VOCAB_SIZE = 10000  # Size of the dictionary
SAVE_PATH = Path("../processed_data.pt")  # Saves to data_pipeline/processed_data.pt


class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_size = vocab_size

    def fit(self, texts):
        print("📊 Building vocabulary...")
        word_counts = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency and take top N words
        sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)

        for word, _ in sorted_words[:self.vocab_size - 2]:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

        print(f"✅ Vocabulary built with {len(self.vocab)} words.")

    def encode(self, text):
        tokens = [self.vocab.get(w, 1) for w in text.lower().split()]  # 1 is UNK

        # Padding / Truncating
        if len(tokens) < MAX_LEN:
            tokens = tokens + [0] * (MAX_LEN - len(tokens))  # Pad with 0
        else:
            tokens = tokens[:MAX_LEN]  # Truncate

        return tokens


def clean_text(text):
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def process_data():
    base_path = Path("../raw_data")
    class_1_path = base_path / "appeal_accepted"
    class_0_path = base_path / "appeal_rejected"

    texts = []
    labels = []

    print("📂 Reading files from disk...")

    # Read Class 1 (Accepted)
    if class_1_path.exists():
        files = list(class_1_path.glob("*.txt"))
        print(f"   Found {len(files)} Accepted cases.")
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                texts.append(clean_text(f.read()))
                labels.append(1)

    # Read Class 0 (Rejected)
    if class_0_path.exists():
        files = list(class_0_path.glob("*.txt"))
        print(f"   Found {len(files)} Rejected cases.")
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                texts.append(clean_text(f.read()))
                labels.append(0)

    if len(texts) == 0:
        print("❌ Error: No data found. Run fetch_real_data.py first!")
        return

    # Tokenize
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.fit(texts)

    print("🔢 Converting text to tensors...")
    input_ids = [tokenizer.encode(t) for t in texts]

    # Convert to PyTorch Tensors
    data_tensor = torch.tensor(input_ids, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    # Save to disk
    torch.save({
        "data": data_tensor,
        "labels": label_tensor,
        "vocab": tokenizer.vocab
    }, SAVE_PATH)

    print(f"💾 Saved processed data to {SAVE_PATH}")


if __name__ == "__main__":
    process_data()