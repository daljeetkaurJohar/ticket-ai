import pandas as pd
import json
import os

from sentence_transformers import SentenceTransformer, util


# ======================================
# Fix encoding issues
# ======================================

def clean_text(text):

    if isinstance(text, str):

        text = text.replace("â€“", "-")
        text = text.replace("â€”", "-")
        text = text.replace("â€˜", "'")
        text = text.replace("â€™", "'")
        text = text.replace("â€œ", '"')
        text = text.replace("â€", '"')

    return text


# ======================================
# FAST Semantic Prototype Classifier
# ======================================

class SemanticPrototypeClassifier:

    def __init__(self):

        print("Loading semantic classifier...")

        # Faster model
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-MiniLM-L3-v2",
            device="cpu"
        )

        examples_path = "data/category_examples.json"

        if not os.path.exists(examples_path):

            raise Exception(
                "category_examples.json not found in data folder."
            )

        with open(examples_path, "r", encoding="utf-8") as f:

            self.examples = json.load(f)


        # Flatten examples
        self.example_texts = []
        self.example_labels = []

        for category, texts in self.examples.items():

            for text in texts:

                self.example_texts.append(text)

                self.example_labels.append(category)


        # Encode example embeddings ONCE
        self.example_embeddings = self.model.encode(
            self.example_texts,
            batch_size=32,
            show_progress_bar=False
        )

        print("Classifier ready.")


    # ======================================
    # Batch prediction (FAST)
    # ======================================

    def predict_batch(self, df):

        texts = df.apply(

            lambda row: " | ".join(
                [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
            ),

            axis=1

        ).tolist()


        # Encode ALL tickets at once
        ticket_embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False
        )


        predicted = []
        confidences = []

        for emb in ticket_embeddings:

            scores = util.cos_sim(
                emb,
                self.example_embeddings
            )[0]

            best_idx = scores.argmax().item()

            predicted.append(
                self.example_labels[best_idx]
            )

            confidences.append(
                float(scores[best_idx])
            )


        return predicted, confidences


# ======================================
# Main classification function
# ======================================

def classify_file(input_file, output_file):

    clf = SemanticPrototypeClassifier()

    df = pd.read_excel(input_file)

    # Fix column names
    df.columns = df.columns.str.strip()

    # FAST batch classification
    predicted, confidences = clf.predict_batch(df)

    df["Predicted Category"] = predicted

    df["Confidence"] = confidences


    # Fix encoding
    for col in df.columns:

        df[col] = df[col].astype(str).apply(clean_text)


    df.to_excel(
        output_file,
        index=False,
        engine="openpyxl"
    )

    print("Classification complete:", output_file)
