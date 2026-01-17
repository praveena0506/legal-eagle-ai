import torch
import os
import re
from pathlib import Path
from collections import Counter

# --- PATHS ---
BASE_DIR = Path(__file__).resolve().parent.parent
ACCEPTED_DIR = BASE_DIR / "raw_data" / "appeal_accepted"
REJECTED_DIR = BASE_DIR / "raw_data" / "appeal_rejected"
PROCESSED_FILE = BASE_DIR / "processed_data.pt"


# 👇 THIS IS THE MISSING FUNCTION
def clean_text(text):
    """
    Simple text cleaning: Lowercase and remove special characters.
    """
    text = str(text).lower()
    # Keep only letters and numbers
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
            # We use the same cleaning logic here
            cleaned = clean_text(text)
            tokens = cleaned.split()
            all_words.extend(tokens)

        # Keep top N most common words
        most_common = Counter(all_words).most_common(self.vocab_size - 2)

        for idx, (word, _) in enumerate(most_common):
            self.vocab[word] = idx + 2  # Start after PAD and UNK

        print(f"✅ Vocab built with {len(self.vocab)} words.")

    def encode(self, text, max_len=512):
        cleaned = clean_text(text)
        tokens = cleaned.split()
        token_ids = [self.vocab.get(t, 1) for t in tokens[:max_len]]

        # Padding
        if len(token_ids) < max_len:
            token_ids += [0] * (max_len - len(token_ids))

        return token_ids


def run_data_preparation():
    print("⚙️ Starting Data Preparation...")

    texts = []
    labels = []

    # 1. Read Accepted Cases (Label = 1)
    if ACCEPTED_DIR.exists():
        for f in ACCEPTED_DIR.glob("*.txt"):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    texts.append(file.read())
                    labels.append(1)
            except Exception:
                pass  # Skip bad files

    # 2. Read Rejected Cases (Label = 0)
    if REJECTED_DIR.exists():
        for f in REJECTED_DIR.glob("*.txt"):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    texts.append(file.read())
                    labels.append(0)
            except Exception:
                pass

    if not texts:
        print("❌ No data found in raw_data folders! Run the Scraper/Auto-Labeler first.")
        return

    # 3. Tokenize
    tokenizer = SimpleTokenizer(vocab_size=2000)
    tokenizer.build_vocab(texts)

    input_ids = []
    for text in texts:
        input_ids.append(tokenizer.encode(text))

    # 4. Save
    data_bundle = {
        "vocab": tokenizer.vocab,
        "data": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }

    torch.save(data_bundle, PROCESSED_FILE)
    print(f"💾 Saved processed data to {PROCESSED_FILE}")


if __name__ == "__main__":
    run_data_preparation()