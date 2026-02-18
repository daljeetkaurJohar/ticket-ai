
import pandas as pd
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"

CATEGORIES = [
"IT - System linkage issue",
"IT - System Access issue",
"IT – System Version issue",
"IT – Data entry handholding",
"IT – Master Data/ mapping issue",
"User - Mapping missing",
"User – Master data delayed input",
"User - Logic changes during ABP",
"User – Master data incorporation in system",
"User – System Knowledge Gap",
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

    categories = []
    confidences = []

    for _, row in df.iterrows():
        cat, conf = clf.predict(row)
        categories.append(cat)
        confidences.append(conf)

    df["Predicted Category"] = categories
    df["Confidence"] = confidences

    df.to_excel(output_file, index=False)

    print("Classification complete:", output_file)
