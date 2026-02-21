import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load model once
model_path = "model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

label_encoder = joblib.load("label_encoder.pkl")

model.eval()

def categorize_ticket(short_desc, desc):

    text = str(short_desc) + " " + str(desc)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][predicted_class].item()

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label, round(confidence, 3)
