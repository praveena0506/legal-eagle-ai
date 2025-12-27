import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformer import TransformerClassifier, create_padding_mask

# --- CONFIGURATION ---
# If True, we generate fake data just to test the code runs.
# If False, you need real data (we will do this in Phase 4).
DEBUG_MODE = True

# Hyperparameters
BATCH_SIZE = 2
D_MODEL = 64  # Small for testing (Real model would be 512)
HEADS = 2  # Small for testing
N_LAYERS = 2
VOCAB_SIZE = 1000  # Small vocab
SEQ_LEN = 20  # Short sentences
NUM_CLASSES = 2  # Guilty (1) vs Innocent (0)
EPOCHS = 1  # Just one pass to prove it works


# --- 1. Create Dummy Data (For Local Testing) ---
class MockLegalDataset(Dataset):
    def __init__(self, num_samples=10):
        # Generate random numbers (simulating words)
        self.data = torch.randint(0, VOCAB_SIZE, (num_samples, SEQ_LEN))
        # Generate random labels (0 or 1)
        self.labels = torch.randint(0, NUM_CLASSES, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train():
    # 1. Setup Device (CPU is fine for 10 samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Training on: {device}")

    # 2. Prepare Data
    if DEBUG_MODE:
        print("⚠️ DEBUG MODE: Using Fake Data")
        dataset = MockLegalDataset(num_samples=10)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    else:
        # Later we will load real legal PDFs here
        pass

    # 3. Initialize Model
    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        h=HEADS,
        d_ff=D_MODEL * 4,
        N=N_LAYERS,
        num_classes=NUM_CLASSES
    ).to(device)

    # 4. Setup Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        print(f"\n--- Epoch {epoch + 1} ---")

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids, labels = input_ids.to(device), labels.to(device)

            # Create mask (important!)
            # 0 is usually the padding index, but in mock data we just use 0
            mask = create_padding_mask(input_ids, pad_token_id=0).to(device)

            # Forward Pass
            optimizer.zero_grad()
            outputs = model(input_ids, mask)

            # Calculate Loss
            loss = criterion(outputs, labels)

            # Backward Pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")

    print("\n✅ Success! The model trained without crashing.")


if __name__ == "__main__":
    train()