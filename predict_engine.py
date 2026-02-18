import pandas as pd
import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer, util


# ==========================
# Clean text
# ==========================

def clean_text(text):

    if isinstance(text, str):

        text = text.replace("â€“", "-")
        text = text.replace("â€”", "-")
        text = text.replace("â€˜", "'")
        text = text.replace("â€™", "'")

    return text


# ==========================
# Enhanced Semantic Classifier
# ==========================

class EnterpriseSemanticClassifier:

    def __init__(self):

        print("Loading enterprise classifier...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        path = "data/category_examples.json"

        if not os.path.exists(path):

            raise Exception("category_examples.json missing")

        with open(path, "r", encoding="utf-8") as f:

            self.examples = json.load(f)

        # Compute centroid per category
        self.category_centroids = {}

        for category, texts in self.examples.items():

            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False
            )

            centroid = np.mean(embeddings, axis=0)

            self.category_centroids[category] = centroid

        print("Classifier ready.")


    def build_context(self, row):

        important_columns = [
            "Ticket Summary",
            "Ticket Details",
            "Problem",
            "Cause",
            "Work notes",
            "Assignment Group",
            "Team"
        ]

        parts = []

        for col in row.index:

            value = row[col]

            if pd.notna(value):

                parts.append(str(value))

        return " | ".join(parts)


    def predict_batch(self, df):

        contexts = df.apply(
            lambda row: self.build_context(row),
            axis=1
        ).tolist()

        ticket_embeddings = self.model.encode(
            contexts,
            batch_size=32,
            show_progress_bar=False
        )

        categories = list(self.category_centroids.keys())

        centroid_matrix = np.vstack(
            [self.category_centroids[c] for c in categories]
        )

        predicted = []
        confidence = []

        for emb in ticket_embeddings:

            scores = util.cos_sim(emb, centroid_matrix)[0]

            idx = scores.argmax().item()

            predicted.append(categories[idx])

            confidence.append(float(scores[idx]))

        return predicted, confidence


# ==========================
# Main function
# ==========================

def classify_file(input_file, output_file):

    clf = EnterpriseSemanticClassifier()

    df = pd.read_excel(input_file)

    df.columns = df.columns.str.strip()

    predicted, conf = clf.predict_batch(df)

    df["Predicted Category"] = predicted

    df["Confidence"] = conf

    df.to_excel(output_file, index=False)

    print("Classification complete.")
