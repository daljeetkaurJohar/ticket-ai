import pandas as pd
from sentence_transformers import SentenceTransformer, util


# Fix encoding function
def clean_text(text):
    if isinstance(text, str):
        text = text.replace("â€“", "-")
        text = text.replace("â€”", "-")
        text = text.replace("â€˜", "'")
        text = text.replace("â€™", "'")
        text = text.replace("â€œ", '"')
        text = text.replace("â€", '"')
    return text


MODEL_NAME = "all-MiniLM-L6-v2"

CATEGORIES = [
    "IT - System linkage issue",
    "IT - System Access issue",
    "IT - System Version issue",
    "IT - Data entry handholding",
    "IT - Master Data/ mapping issue",
    "User - Mapping missing",
    "User - Master data delayed input",
    "User - Logic changes during ABP",
    "User - Master data incorporation in system",
    "User - System Knowledge Gap",
    "User - Logic mistakes in excel vs system",
    "User - Multiple versions issue in excel"
]


class OfflineClassifier:

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.cat_embeddings = self.model.encode(CATEGORIES)

    def build_context(self, row):
        return " | ".join([f"{col}: {row[col]}" for col in row.index])

    def predict(self, row):
        text = self.build_context(row)
        emb = self.model.encode(text)
        scores = util.cos_sim(emb, self.cat_embeddings)
        idx = scores.argmax()
        conf = float(scores.max())
        return CATEGORIES[idx], conf


def classify_file(input_file, output_file):

    clf = OfflineClassifier()

    df = pd.read_excel(input_file)

    # Remove hidden spaces from column names
    df.columns = df.columns.str.strip()

    categories = []
    confidences = []

    for _, row in df.iterrows():

        cat, conf = clf.predict(row)

        categories.append(cat)
        confidences.append(conf)

    # Assign predictions
    df["Predicted Category"] = categories
    df["Confidence"] = confidences

    # Fix encoding issues
    for col in df.columns:
        df[col] = df[col].astype(str).apply(clean_text)

    # Save file
    df.to_excel(output_file, index=False, engine="openpyxl")

    print("Classification complete:", output_file)
