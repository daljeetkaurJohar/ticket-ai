import pandas as pd
import numpy as np
import json
import os

from sentence_transformers import SentenceTransformer, util
from classifier import auto_label  # rule engine boost


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
JSON_FILE = "data/category_examples.json"
AUTO_LEARN_FILE = "data/auto_learn.xlsx"
CONFIDENCE_THRESHOLD = 0.85


class EnterpriseIntentClassifier:

    def __init__(self):

        print("Loading Enterprise Intent Classifier...")

        if not os.path.exists(JSON_FILE):
            raise Exception("category_examples.json not found in data folder")

        self.model = SentenceTransformer(MODEL_NAME)

        with open(JSON_FILE, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.categories = []
        centroid_list = []

        for category, content in self.data.items():

            vectors = []

            if isinstance(content, list):
                vectors.extend(self.model.encode(content))
            else:
                if "examples" in content:
                    vectors.extend(self.model.encode(content["examples"]))
                if "symptoms" in content:
                    vectors.extend(self.model.encode(content["symptoms"]))
                if "causes" in content:
                    vectors.extend(self.model.encode(content["causes"]))
                if "definition" in content:
                    vectors.extend(self.model.encode([content["definition"]]))

            centroid = np.mean(vectors, axis=0)
            centroid_list.append(centroid)
            self.categories.append(category)

        self.centroids = np.vstack(centroid_list)

        print("Classifier ready with", len(self.categories), "categories")

    # ==========================
    # Weighted context builder
    # ==========================
    def build_context(self, row):

        weights = {
            "Ticket Summary": 2,
            "Ticket Details": 3,
            "Solution": 3,
            "Work notes": 2,
            "Work Notes": 2
        }

        parts = []

        for col, weight in weights.items():
            if col in row and pd.notna(row[col]):
                parts.extend([str(row[col])] * weight)

        return " | ".join(parts)

    # ==========================
    # Hybrid Prediction
    # ==========================
    def predict_batch(self, df):

        contexts = df.apply(self.build_context, axis=1).tolist()

        embeddings = self.model.encode(
            contexts,
            batch_size=32,
            show_progress_bar=False
        )

        predicted = []
        confidence = []

        for i, emb in enumerate(embeddings):

            scores = util.cos_sim(
                emb,
                self.centroids
            )[0].cpu().numpy()

            best_idx = np.argmax(scores)
            best_score = float(scores[best_idx])

            ml_prediction = self.categories[best_idx]

            # Rule engine boost
            rule_prediction = auto_label(contexts[i])

            if rule_prediction == ml_prediction:
                best_score += 0.05  # small boost

            predicted.append(ml_prediction)
            confidence.append(round(best_score, 3))

        return predicted, confidence

    # ==========================
    # Auto learning
    # ==========================
    def auto_learn(self, df):

        high_conf = df[df["Confidence"] > 0.80]

        if len(high_conf) == 0:
            return

        if os.path.exists(AUTO_LEARN_FILE):
            old = pd.read_excel(AUTO_LEARN_FILE)
            combined = pd.concat([old, high_conf])
        else:
            combined = high_conf

        combined.to_excel(AUTO_LEARN_FILE, index=False)

        print("Auto-learning data updated.")


def classify_file(input_file, output_file):

    clf = EnterpriseIntentClassifier()

    df = pd.read_excel(input_file)
    df.columns = df.columns.str.strip()

    pred, conf = clf.predict_batch(df)

    df["Predicted Category"] = pred
    df["Confidence"] = conf

    clf.auto_learn(df)

    df.to_excel(output_file, index=False)

    print("Classification complete.")
