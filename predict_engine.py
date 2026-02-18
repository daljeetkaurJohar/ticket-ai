import pandas as pd
import json
import os

from sentence_transformers import SentenceTransformer, util


# ================================
# Fix encoding issues
# ================================

def clean_text(text):

    if isinstance(text, str):

        text = text.replace("â€“", "-")
        text = text.replace("â€”", "-")
        text = text.replace("â€˜", "'")
        text = text.replace("â€™", "'")
        text = text.replace("â€œ", '"')
        text = text.replace("â€", '"')

    return text


# ================================
# Semantic Prototype Classifier
# ================================

class SemanticPrototypeClassifier:

    def __init__(self):

        print("Loading semantic prototype classifier...")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        examples_path = "data/category_examples.json"

        if not os.path.exists(examples_path):

            raise Exception(
                "category_examples.json not found. Run generate_examples.py first."
            )

        with open(examples_path, "r", encoding="utf-8") as f:

            self.examples = json.load(f)

        self.example_embeddings = {}

        for category, texts in self.examples.items():

            if texts:

                self.example_embeddings[category] = self.model.encode(texts)

        print("Classifier ready.")


    def build_context(self, row):

        parts = []

        for col in row.index:

            value = row[col]

            if pd.notna(value):

                parts.append(f"{col}: {value}")

        return " | ".join(parts)


    def predict(self, row):

        text = self.build_context(row)

        ticket_embedding = self.model.encode(text)

        best_category = None

        best_score = -1

        for category, embeddings in self.example_embeddings.items():

            scores = util.cos_sim(ticket_embedding, embeddings)

            score = float(scores.max())

            if score > best_score:

                best_score = score

                best_category = category

        return best_category, best_score


# ================================
# Main classification function
# ================================

def classify_file(input_file, output_file):

    clf = SemanticPrototypeClassifier()

    df = pd.read_excel(input_file)

    df.columns = df.columns.str.strip()

    predicted_categories = []
    confidences = []

    for _, row in df.iterrows():

        category, confidence = clf.predict(row)

        predicted_categories.append(category)
        confidences.append(confidence)

    df["Predicted Category"] = predicted_categories

    df["Confidence"] = confidences

    for col in df.columns:

        df[col] = df[col].astype(str).apply(clean_text)

    df.to_excel(output_file, index=False, engine="openpyxl")

    print("Classification complete:", output_file)
