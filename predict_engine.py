import pandas as pd
import json
import os

from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader


MODEL_PATH = "model/fine_tuned"
LABEL_MAP_PATH = "model/label_map.json"


# ==========================
# Auto training function
# ==========================

def train_model_if_missing():

    if os.path.exists(MODEL_PATH):

        print("Model already exists.")
        return


    print("Training model automatically...")

    os.makedirs("model", exist_ok=True)

    df = pd.read_excel("data/incoming_tickets.xlsx")

    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["ISSUE CAT"])


    def build_text(row):

        parts = []

        cols = [
            "Ticket Summary",
            "Ticket Details",
            "Problem",
            "Cause"
        ]

        for col in cols:

            if col in row and pd.notna(row[col]):

                parts.append(str(row[col]))

        return " | ".join(parts)


    df["text"] = df.apply(build_text, axis=1)


    categories = df["ISSUE CAT"].unique()

    label_map = {cat: i for i, cat in enumerate(categories)}

    with open(LABEL_MAP_PATH, "w") as f:

        json.dump(label_map, f)


    train_examples = []

    for _, row in df.iterrows():

        train_examples.append(
            InputExample(texts=[row["text"], row["ISSUE CAT"]])
        )


    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


    dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=16
    )


    loss = losses.CosineSimilarityLoss(model)


    model.fit(

        train_objectives=[(dataloader, loss)],

        epochs=3,

        warmup_steps=100
    )


    model.save(MODEL_PATH)

    print("Model trained and saved.")


# ==========================
# Classifier
# ==========================

class AutoClassifier:

    def __init__(self):

        train_model_if_missing()

        self.model = SentenceTransformer(MODEL_PATH)

        with open(LABEL_MAP_PATH) as f:

            label_map = json.load(f)

        self.categories = list(label_map.keys())

        self.cat_embeddings = self.model.encode(
            self.categories
        )


    def build_context(self, row):

        parts = []

        for col in row.index:

            if pd.notna(row[col]):

                parts.append(str(row[col]))

        return " | ".join(parts)


    def predict_batch(self, df):

        texts = df.apply(
            lambda row: self.build_context(row),
            axis=1
        ).tolist()

        embeddings = self.model.encode(
            texts,
            batch_size=32
        )


        predicted = []
        confidence = []

        for emb in embeddings:

            scores = util.cos_sim(
                emb,
                self.cat_embeddings
            )[0]

            idx = scores.argmax().item()

            predicted.append(
                self.categories[idx]
            )

            confidence.append(
                float(scores[idx])
            )

        return predicted, confidence


# ==========================
# Main function
# ==========================

def classify_file(input_file, output_file):

    clf = AutoClassifier()

    df = pd.read_excel(input_file)

    df.columns = df.columns.str.strip()

    pred, conf = clf.predict_batch(df)

    df["Predicted Category"] = pred

    df["Confidence"] = conf

    df.to_excel(output_file, index=False)

    print("Classification complete.")
