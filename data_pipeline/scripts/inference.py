import torch
import re
import sys
import os
from pathlib import Path

# --- SMART PATH CONFIGURATION ---
# Get the location of THIS script (inference.py)
current_script_path = Path(__file__).resolve().parent

# Go up 2 levels (scripts -> data_pipeline -> legal-eagle-ai) then down into models
MODEL_PATH = current_script_path.parents[1] / "models" / "legal_eagle_model.pth"

# Add current directory to path for imports
sys.path.append(str(current_script_path))

print(f"🔍 Looking for model at: {MODEL_PATH}")

from transformer import TransformerClassifier, create_padding_mask
from prepare_data import SimpleTokenizer, clean_text
# ... (Rest of the class LegalAI code remains the same) ...

# --- CONFIGURATION ---

class LegalAI:
    def __init__(self):
        print("🧠 Loading Legal Eagle AI...")

        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}. Did you download it from Colab?")

        # 1. Load the checkpoint
        # map_location='cpu' ensures it works on your laptop even without a GPU
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

        # 2. Rebuild the Tokenizer (using the vocab saved in the file)
        self.vocab = checkpoint.get('vocab')
        if not self.vocab:
            # Fallback if vocab wasn't saved inside the dictionary (older versions)
            print("⚠️ Warning: Vocab not found in checkpoint. Using default.")
            self.vocab = {"<PAD>": 0, "<UNK>": 1}

        self.tokenizer = SimpleTokenizer()
        self.tokenizer.vocab = self.vocab

        # 3. Rebuild the Brain (Model Architecture)
        # These numbers MUST match what we trained with in Colab (64, 4, 256, 2)
        self.model = TransformerClassifier(
            vocab_size=len(self.vocab),
            d_model=64,
            h=4,
            d_ff=256,
            N=2,
            num_classes=2
        )

        # 4. Load the learned weights
        # We use strict=False in case there are minor key mismatches, but usually it's fine
        self.model.load_state_dict(checkpoint if 'state_dict' not in checkpoint else checkpoint['state_dict'])
        self.model.eval()  # Set to evaluation mode (turns off dropout)
        print("✅ AI is ready to judge!")

    def predict(self, text):
        # Clean
        text = clean_text(text)

        # Tokenize
        tokens = self.tokenizer.encode(text)  # Uses the internal MAX_LEN of the tokenizer

        # Convert to Tensor
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        mask = create_padding_mask(input_tensor, pad_token_id=0)

        # Predict
        with torch.no_grad():
            output = self.model(input_tensor, mask)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        verdict = "ACCEPTED (Petitioner Won)" if predicted_class.item() == 1 else "REJECTED (Respondent Won)"
        return verdict, confidence.item()


# --- TEST AREA ---
if __name__ == "__main__":
    ai = LegalAI()

    print("\n--- ⚖️  TESTING YOUR MODEL LOCALLY ---")

    # Test Case 1 (Winning Language)
    text1 = "The appeal is allowed. The high court's decision is set aside."
    print(f"\n📝 Input: {text1}")
    result, conf = ai.predict(text1)
    print(f"🤖 Verdict: {result} (Confidence: {conf:.2f})")

    # Test Case 2 (Losing Language)
    text2 = "The appeal is dismissed. We find no merit in the arguments."
    print(f"\n📝 Input: {text2}")
    result, conf = ai.predict(text2)
    print(f"🤖 Verdict: {result} (Confidence: {conf:.2f})")