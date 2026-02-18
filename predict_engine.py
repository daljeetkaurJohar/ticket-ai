import pandas as pd
import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer, util


# =====================================
# Clean text
# =====================================

def clean_text(text):

    if isinstance(text, str):

        text = text.replace("â€“", "-")
        text = text.replace("â€”", "-")
        text = text.replace("â€˜", "'")
        text = text.replace("â€™", "'")
        text = text.replace("â€œ", '"')
        text = text.replace("â€", '"')

    return text


# =====================================
# Fine-tuned classifier
# =====================================

class FineTunedClassifier:

    def __init__(self):

        print("Loading fine-tuned model...")

        model_path = "model/fine_tuned"

        if not os.path.exists(model_path):

            raise Exception(
                "Fine-tuned model not found. Run train_classifier.py first."
            )

        self.model = SentenceTransformer(model_path)

        # Load label map
        with open("model/label_map.json", "r") as f:

            self.label_map = json.load(f)

        self.reverse_map = {

            int(v): k for k, v in self.label_map.items()

        }

        # Create label embeddings
        self.label_texts = list(self.label_map.keys())

        self.label_embeddings = self.model.encode(
            self.label_texts,
            batch_size=32,
            show_progress_bar=False
        )

        print("Fine-tuned classifier ready.")


    # =====================================
    # Build context from ALL fields
    # =====================================

    def build_context(self, row):

        important_columns = [
            "Ticket Summary",
            "Ticket Details",
            "Problem",
            "Cause",
            "Assignment Group",
            "Work notes"
        ]

        parts = []

        for col in important_columns:

            if col in row and pd.notna(row[col]):

                parts.append(str(row[col]))

        return " | ".join(parts)


    # =====================================
    # Batch prediction
    # =====================================

    def predict_batch(self, df):

        texts = df.apply(
            lambda row: self.build_context(row),
            axis=1
        ).tolist()

        ticket_embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False
        )

        predicted = []
        confidence = []

        for emb in ticket_embeddings:

            scores = util.cos_sim(
                emb,
                self.label_embeddings
            )[0]

            idx = scores.argmax().item()

            predicted.append(self.label_texts[idx])

            confidence.append(float(scores[idx]))

        return predicted, confidence


# =====================================
# Main function used by dashboard
# =====================================

def classify_file(input_file, output_file):

    clf = FineTunedClassifier()

    df = pd.read_excel(input_file)

    df.columns = df.columns.str.strip()

    predicted, confidence = clf.predict_batch(df)

    df["Predicted Category"] = predicted

    df["Confidence"] = confidence

    # Clean encoding
    for col in df.columns:

        df[col] = df[col].astype(str).apply(clean_text)

    df.to_excel(
        output_file,
        index=False,
        engine="openpyxl"
    )

    print("Classification complete:", output_file)
