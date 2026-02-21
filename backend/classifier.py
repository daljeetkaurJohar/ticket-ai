# backend/classifier.py

import torch
import joblib
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from categorization_logic import CategorizationLogic

# =============================
# Load Excel Logic
# =============================
logic = CategorizationLogic("data/issue category.xlsx")

# =============================
# Load ML Model
# =============================
MODEL_PATH = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
label_encoder = joblib.load("label_encoder.pkl")

model.eval()

# =============================
# Hybrid Prediction
# =============================

def predict_ticket(text: str):

    # Step 1: Excel Rule Logic
    rule_category, rule_conf = logic.categorize(text)

    if rule_category != "Needs Manual Review":
        return {
            "category": rule_category,
            "confidence": rule_conf,
            "source": "Excel Logic"
        }

    # Step 2: ML Fallback
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

    label = label_encoder.inverse_transform([pred_class])[0]

    return {
        "category": label,
        "confidence": round(confidence, 3),
        "source": "ML"
    }
