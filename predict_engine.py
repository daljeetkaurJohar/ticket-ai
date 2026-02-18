import pandas as pd
import numpy as np
import os

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CONFIDENCE_THRESHOLD = 0.55


class EnterpriseHybridClassifier:

    def __init__(self):

        print("Loading enterprise hybrid classifier...")

        self.model = SentenceTransformer(MODEL_NAME)

        df = pd.read_excel("data/incoming_tickets.xlsx")

        df.columns = df.columns.str.strip()

        df = df.dropna(subset=["ISSUE CAT"])


        # Build strong semantic context
        def build_context(row):

            priority_cols = [
                "Ticket Summary",
                "Ticket Details",
                "Problem",
                "Cause",
                "Assignment Group",
                "Work notes"
            ]

            parts = []

            for col in priority_cols:

                if col in row and pd.notna(row[col]):

                    parts.append(str(row[col]))

            return " | ".join(parts)


        df["context"] = df.apply(build_context, axis=1)


        # Create centroids
        self.categories = []
        centroid_list = []

        for category, group in df.groupby("ISSUE CAT"):

            texts = group["context"].tolist()

            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False
            )

            centroid = np.mean(embeddings, axis=0)

            centroid_list.append(centroid)
            self.categories.append(category)

        self.centroids = np.vstack(centroid_list)

        print("Hybrid classifier ready.")


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

            scores = util.cos_sim(
                emb,
                self.centroids
            )[0]

            scores_np = scores.cpu().numpy()

            best_idx = np.argmax(scores_np)
            best_score = scores_np[best_idx]

            # Secondary match
            sorted_idx = np.argsort(scores_np)[::-1]

            second_idx = sorted_idx[1]
            second_score = scores_np[second_idx]


            # Hybrid decision logic
            if best_score < CONFIDENCE_THRESHOLD:

                category = "Uncertain"

            elif best_score - second_score < 0.05:

                # choose safer category
                category = self.categories[second_idx]

            else:

                category = self.categories[best_idx]


            predicted.append(category)
            confidence.append(float(best_score))


        return predicted, confidence


def classify_file(input_file, output_file):

    clf = EnterpriseHybridClassifier()

    df = pd.read_excel(input_file)

    df.columns = df.columns.str.strip()

    pred, conf = clf.predict_batch(df)

    df["Predicted Category"] = pred

    df["Confidence"] = conf

    df.to_excel(output_file, index=False)

    print("Enterprise hybrid classification complete.")
