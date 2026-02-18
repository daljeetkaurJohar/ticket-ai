import pandas as pd
import numpy as np
import json
import os

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
JSON_FILE = "data/category_examples.json"


class JSONClassifier:

    def __init__(self):

        print("Loading classifier from JSON examples...")

        if not os.path.exists(JSON_FILE):

            raise Exception("category_examples.json not found in data folder")

        self.model = SentenceTransformer(MODEL_NAME)

        with open(JSON_FILE, "r", encoding="utf-8") as f:

            self.examples = json.load(f)


        self.categories = []
        centroid_list = []


        # Create centroid per category from JSON examples
        for category, texts in self.examples.items():

            embeddings = self.model.encode(
                texts,
                batch_size=16,
                show_progress_bar=False
            )

            centroid = np.mean(embeddings, axis=0)

            centroid_list.append(centroid)
            self.categories.append(category)


        self.centroids = np.vstack(centroid_list)

        print("JSON classifier ready.")


    def build_context(self, row):

        important_cols = [
            "Ticket Summary",
            "Ticket Details",
            "Problem",
            "Cause",
            "Assignment Group",
            "Work notes"
        ]

        parts = []

        for col in important_cols:

            if col in row and pd.notna(row[col]):

                parts.append(str(row[col]))

        return " | ".join(parts)


    def predict_batch(self, df):

        contexts = df.apply(self.build_context, axis=1).tolist()

        embeddings = self.model.encode(
            contexts,
            batch_size=32,
            show_progress_bar=False
        )

        predicted = []
        confidence = []

        for emb in embeddings:

            scores = util.cos_sim(
                emb,
                self.centroids
            )[0].cpu().numpy()

            idx = np.argmax(scores)

            predicted.append(self.categories[idx])
            confidence.append(float(scores[idx]))

        return predicted, confidence


def classify_file(input_file, output_file):

    clf = JSONClassifier()

    df = pd.read_excel(input_file)

    df.columns = df.columns.str.strip()

    pred, conf = clf.predict_batch(df)

    df["Predicted Category"] = pred
    df["Confidence"] = conf

    df.to_excel(output_file, index=False)

    print("Classification complete.")
