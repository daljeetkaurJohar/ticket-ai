import torch
import joblib
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model once
MODEL_PATH = "model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
label_encoder = joblib.load("label_encoder.pkl")

model.eval()

def predict_ticket(text: str):
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
        "confidence": round(confidence, 3)
    }
