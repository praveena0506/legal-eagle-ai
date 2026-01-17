import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
import os

# Fix import path for transformer
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "scripts"))

from transformer import TransformerClassifier, create_padding_mask

# Paths
PROCESSED_FILE = BASE_DIR / "processed_data.pt"
MODEL_SAVE_DIR = BASE_DIR.parent / "models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_SAVE_DIR / "legal_eagle_model.pth"


def run_training_loop():
    print("💪 Starting Training Loop...")

    if not PROCESSED_FILE.exists():
        print("❌ processed_data.pt not found. Run preparation first.")
        return

    # 1. Load Data
    saved_data = torch.load(PROCESSED_FILE)
    vocab = saved_data["vocab"]
    dataset = TensorDataset(saved_data["data"], saved_data["labels"])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"   Loaded {len(dataset)} training samples.")

    # 2. Initialize Model
    device = torch.device("cpu")  # Docker usually runs on CPU
    model = TransformerClassifier(vocab_size=len(vocab), d_model=64, num_classes=2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Train
    model.train()
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for batch_ids, batch_labels in dataloader:
            batch_ids, batch_labels = batch_ids.to(device), batch_labels.to(device)
            mask = create_padding_mask(batch_ids)

            optimizer.zero_grad()
            outputs = model(batch_ids, mask)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"   Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")

    # 4. Save Model with Vocab (Important!)
    checkpoint = {
        "state_dict": model.state_dict(),
        "vocab": vocab
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"🎉 Model updated and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    run_training_loop()