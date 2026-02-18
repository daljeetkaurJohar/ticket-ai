import pandas as pd
import numpy as np
import json

from sentence_transformers import SentenceTransformer, util


MODEL = "sentence-transformers/all-MiniLM-L6-v2"
JSON_FILE = "data/category_examples.json"


class EnterpriseIntentClassifier:

    def __init__(self):

        print("Loading enterprise intent classifier...")

        self.model = SentenceTransformer(MODEL)

        with open(JSON_FILE, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.categories = []
        centroid_list = []

        for category, content in self.data.items():

            # Combine all semantic info
            texts = []

            if isinstance(content, dict):

                texts.append(content.get("definition", ""))

                texts.extend(content.get("symptoms", []))

                texts.extend(content.get("causes", []))

                texts.extend(content.get("examples", []))

            else:
                texts.extend(content)

            embeddings = self.model.encode(texts)

            centroid = np.mean(embeddings, axis=0)

            centroid_list.append(centroid)

            self.categories.append(category)

        self.centroids = np.vstack(centroid_list)


    def build_context(self, row):

        weights = {
            "Ticket Summary": 3,
            "Ticket Details": 3,
            "Problem": 2,
            "Cause": 2,
            "Work notes": 1,
            "Assignment Group": 1
        }

        parts = []

        for col, weight in weights.items():

            if col in row and pd.notna(row[col]):

                parts.extend([str(row[col])] * weight)

        return " | ".join(parts)


    def predict_batch(self, df):

        contexts = df.apply(self.build_context, axis=1).tolist()

        embeddings = self.model.encode(contexts)

        predicted = []
        confidence = []

        for emb in embeddings:

            scores = util.cos_sim(emb, self.centroids)[0]

            idx = scores.argmax().item()

            predicted.append(self.categories[idx])

            confidence.append(float(scores[idx]))

        return predicted, confidence


def classify_file(input_file, output_file):

    clf = EnterpriseIntentClassifier()

    df = pd.read_excel(input_file)

    pred, conf = clf.predict_batch(df)

    df["Predicted Category"] = pred

    df["Confidence"] = conf

    df.to_excel(output_file, index=False)

    print("Classification complete.")
