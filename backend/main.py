from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from pathlib import Path

# --- SETUP PATHS ---
# This trick finds the 'legal-eagle-ai' folder automatically
# so we can import 'inference.py' from the data_pipeline folder.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "data_pipeline" / "scripts"))

# Import your AI Logic
try:
    from inference import LegalAI
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not import LegalAI. Check paths: {e}")
    sys.exit(1)

# --- INITIALIZE APP ---
app = FastAPI(
    title="Legal Eagle AI ⚖️",
    description="An AI API that predicts if a petitioner will Win or Lose.",
    version="1.0"
)

# --- LOAD MODEL (Global Variable) ---
# We load the model ONCE when the server starts
print("⏳ Booting up the AI Model...")
try:
    ai_engine = LegalAI()
    print("✅ AI System Online & Ready!")
except Exception as e:
    print(f"❌ Failed to load AI: {e}")
    ai_engine = None


# --- DEFINE INPUT DATA ---
class CaseRequest(BaseModel):
    text: str


# --- API ENDPOINTS ---

@app.get("/")
def home():
    """Health Check: Checks if the server is running."""
    return {"status": "online", "message": "Legal Eagle AI is ready to judge."}


@app.post("/predict")
def predict_verdict(request: CaseRequest):
    """
    Takes legal text -> Returns Verdict & Confidence.
    """
    if not ai_engine:
        raise HTTPException(status_code=500, detail="AI Model is not loaded.")

    # Ask the AI
    verdict, confidence = ai_engine.predict(request.text)

    return {
        "verdict": verdict,
        "confidence_score": f"{confidence:.2%}",  # e.g. "98.50%"
        "input_snippet": request.text[:100] + "..."
    }