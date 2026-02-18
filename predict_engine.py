import pandas as pd
import json
import os
import numpy as np

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EnterpriseClassifier:

    def __init__(self):

        print("Loading enterprise classifier...")

        self.model = SentenceTransformer(MODEL_NAME)

        df = pd.read_excel("data/incoming_tickets.xlsx")

        df.columns = df.columns.str.strip()

        df = df.dropna(subset=["ISSUE CAT"])


        def build_context(row):

            cols = [
                "Ticket Summary",
                "Ticket Details",
                "Problem",
                "Cause",
                "Assignment Group"
            ]

            parts = []

            for col in cols:

                if col in row and pd.notna(row[col]):

                    parts.append(str(row[col]))

            return " | ".join(parts)


        df["context"] = df.apply(build_context, axis=1)


        # Compute centroid embedding per category
        self.categories = []
        self.centroids = []

        for category, group in df.groupby("ISSUE CAT"):

            texts = group["context"].tolist()

            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False
            )

            centroid = np.mean(embeddings, axis=0)

            self.categories.append(category)
            self.centroids.append(centroid)

        self.centroids = np.vstack(self.centroids)

        print("Classifier ready.")


    def predict_batch(self, df):

        def build_context(row):

            parts = []

            for col in row.index:

                if pd.notna(row[col]):

                    parts.append(str(row[col]))

            return " | ".join(parts)


        texts = df.apply(build_context, axis=1).tolist()

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False
        )


        predicted = []
        confidence = []

        for emb in embeddings:

            scores = util.cos_sim(emb, self.centroids)[0]

            idx = scores.argmax().item()

            predicted.append(self.categories[idx])
            confidence.append(float(scores[idx]))

        return predicted, confidence


def classify_file(input_file, output_file):

    clf = EnterpriseClassifier()

    df = pd.read_excel(input_file)

    df.columns = df.columns.str.strip()

    pred, conf = clf.predict_batch(df)

    df["Predicted Category"] = pred

    df["Confidence"] = conf

    df.to_excel(output_file, index=False)

    print("Classification complete.")
