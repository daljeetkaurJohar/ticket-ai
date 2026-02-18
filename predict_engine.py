import pandas as pd
import os
import json
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================
# Fix encoding function
# ============================

def clean_text(text):
    if isinstance(text, str):
        text = text.replace("â€“", "-")
        text = text.replace("â€”", "-")
        text = text.replace("â€˜", "'")
        text = text.replace("â€™", "'")
        text = text.replace("â€œ", '"')
        text = text.replace("â€", '"')
    return text


# ============================
# Categories
# ============================

CATEGORIES = [
    "IT - System linkage issue",
    "IT - System Access issue",
    "IT - System Version issue",
    "IT - Data entry handholding",
    "IT - Master Data/ mapping issue",
    "User - Mapping missing",
    "User - Master data delayed input",
    "User - Logic changes during ABP",
    "User - Master data incorporation in system",
    "User - System Knowledge Gap",
    "User - Logic mistakes in excel vs system",
    "User - Multiple versions issue in excel"
]


# ============================
# Fine-tuned classifier
# ============================

class FineTunedClassifier:

    def __init__(self):

        self.model = AutoModelForSequenceClassification.from_pretrained("model")

        self.tokenizer = AutoTokenizer.from_pretrained("model")

        with open("model/label_map.json") as f:
            self.label_map = json.load(f)

        self.reverse_map = {v: k for k, v in self.label_map.items()}


    def predict(self, row):

        text = " | ".join([f"{col}: {row[col]}" for col in row.index])

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        outputs = self.model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()

        confidence = torch.softmax(outputs.logits, dim=1).max().item()

        return self.reverse_map[pred], confidence


# ============================
# Semantic fallback classifier
# ============================

class OfflineClassifier:

    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.cat_embeddings = self.model.encode(CATEGORIES)


    def predict(self, row):

        text = " | ".join([f"{col}: {row[col]}" for col in row.index])

        emb = self.model.encode(text)

        scores = util.cos_sim(emb, self.cat_embeddings)

        idx = scores.argmax()

        confidence = float(scores.max())

        return CATEGORIES[idx], confidence


# ============================
# Main classification function
# ============================

def classify_file(input_file, output_file):

    # Select classifier automatically
    if os.path.exists("model") and os.path.exists("model/label_map.json"):

        print("Using Fine-Tuned Model")

        clf = FineTunedClassifier()

    else:

        print("Using Semantic Similarity Model")

        clf = OfflineClassifier()


    df = pd.read_excel(input_file)

    # Fix column names
    df.columns = df.columns.str.strip()

    predicted_categories = []
    confidences = []

    for _, row in df.iterrows():

        category, confidence = clf.predict(row)

        predicted_categories.append(category)
        confidences.append(confidence)


    df["Predicted Category"] = predicted_categories
    df["Confidence"] = confidences


    # Fix encoding
    for col in df.columns:

        df[col] = df[col].astype(str).apply(clean_text)


    df.to_excel(output_file, index=False, engine="openpyxl")

    print("Classification complete:", output_file)
