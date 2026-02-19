import json
import torch
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from classifier import rule_override

class EliteClassifier:

    def __init__(self):

        self.mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        with open("models/mpnet/embeddings.json") as f:
            self.category_embeddings = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained("models/deberta")
        self.deberta = AutoModelForSequenceClassification.from_pretrained("models/deberta")

        self.meta = joblib.load("models/meta.pkl")

        with open("models/temperature.pkl", "rb") as f:
            self.temperature = joblib.load(f)

    def predict(self, text):

        # Stage 0 Rule Override
        rule_pred = rule_override(text)
        if rule_pred:
            return rule_pred, 1.0

        # Stage 1 MPNet Retrieval
        emb = self.mpnet.encode(text)

        scores = {}
        for cat, embs in self.category_embeddings.items():
            embs = torch.tensor(embs)
            sim = util.cos_sim(torch.tensor(emb), embs)
            scores[cat] = float(sim.max())

        top_cat = max(scores, key=scores.get)

        # Stage 2 DeBERTa
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.deberta(**inputs)
        probs = torch.softmax(outputs.logits / self.temperature, dim=1)

        tf_score = torch.max(probs).item()

        # Stage 3 Meta Feature
        feature_vector = np.array([[scores[top_cat], tf_score]])

        final_pred = self.meta.predict(feature_vector)[0]

        confidence = float(max(scores[top_cat], tf_score))

        return final_pred, round(confidence, 3)
