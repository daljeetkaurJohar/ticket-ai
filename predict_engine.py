import pandas as pd
import numpy as np
import os

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TRAIN_FILE = "data/training_data.xlsx"

DATA_FOLDER = "data"

CONFIDENCE_THRESHOLD = 0.55

MAX_PER_CATEGORY = 50


class EnterpriseHybridClassifier:

    def __init__(self):

        print("Initializing Enterprise Classifier...")

        self.ensure_training_file()

        if not os.path.exists(TRAIN_FILE):

            raise Exception(
                "No training file found. Please place one Excel file with ISSUE CAT column inside data folder."
            )

        self.model = SentenceTransformer(MODEL_NAME)

        df = pd.read_excel(TRAIN_FILE)

        df.columns = df.columns.str.strip()

        df = df.dropna(subset=["ISSUE CAT"])

        df["context"] = df.apply(self.build_context, axis=1)

        self.categories = []

        centroid_list = []

        # Balanced centroid creation
        for category, group in df.groupby("ISSUE CAT"):

            group = group.sample(
                min(len(group), MAX_PER_CATEGORY),
                random_state=42
            )

            embeddings = self.model.encode(
                group["context"].tolist(),
                batch_size=32,
                show_progress_bar=False
            )

            centroid = np.mean(embeddings, axis=0)

            centroid_list.append(centroid)

            self.categories.append(category)

        self.centroids = np.vstack(centroid_list)

        print("Classifier ready.")


    # Automatically create training_data.xlsx if missing
    def ensure_training_file(self):

        if os.path.exists(TRAIN_FILE):

            print("training_data.xlsx already exists.")

            return

        print("Searching for labeled training file...")

        for file in os.listdir(DATA_FOLDER):

            if file.endswith(".xlsx"):

                path = os.path.join(DATA_FOLDER, file)

                df = pd.read_excel(path)

                df.columns = df.columns.str.strip()

                if "ISSUE CAT" in df.columns:

                    df.to_excel(TRAIN_FILE, index=False)

                    print(f"training_data.xlsx created from {file}")

                    return

        print("No labeled file found in data folder.")


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

        df["context"] = df.apply(self.build_context, axis=1)

        embeddings = self.model.encode(
            df["context"].tolist(),
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

            best_idx = np.argmax(scores)

            best_score = scores[best_idx]

            sorted_idx = np.argsort(scores)[::-1]

            second_idx = sorted_idx[1]

            second_score = scores[second_idx]


            # Hybrid logic
            if best_score < CONFIDENCE_THRESHOLD:

                category = "Uncertain"

            elif best_score - second_score < 0.05:

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

    print("Classification complete.")
