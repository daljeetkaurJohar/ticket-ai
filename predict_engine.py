import json
import torch
import numpy as np
import joblib
import os

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from classifier import rule_override


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EliteClassifier:

    def __init__(self):

        print("Loading Elite Classifier...")

        # ----------------------------
        # MPNet Semantic Model
        # ----------------------------
        self.mpnet = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device=DEVICE
        )

        if not os.path.exists("models/mpnet/embeddings.json"):
            raise Exception("MPNet embeddings not found. Run train_mpnet.py first.")

        with open("models/mpnet/embeddings.json", "r") as f:
            self.category_embeddings = json.load(f)

        # Convert stored embeddings to torch tensors (once)
        self.category_tensors = {
            cat: torch.tensor(embs, dtype=torch.float32, device=DEVICE)
            for cat, embs in self.category_embeddings.items()
        }

        # ----------------------------
        # DeBERTa Model
        # ----------------------------
        if not os.path.exists("models/deberta"):
            raise Exception("DeBERTa model not found. Run train_deberta.py first.")

        self.tokenizer = AutoTokenizer.from_pretrained("models/deberta")
        self.deberta = AutoModelForSequenceClassification.from_pretrained(
            "models/deberta"
        ).to(DEVICE)

        self.deberta.eval()

        # ----------------------------
        # Meta Model (Optional)
        # ----------------------------
        if os.path.exists("models/meta.pkl"):
            self.meta_model = joblib.load("models/meta.pkl")
        else:
            self.meta_model = None

        # ----------------------------
        # Temperature Scaling (Optional)
        # ----------------------------
        if os.path.exists("models/temperature.pkl"):
            self.temperature = joblib.load("models/temperature.pkl")
        else:
            self.temperature = 1.0

        print("Elite Classifier Ready.")

    # ==================================================
    # Multi-Prototype Semantic Similarity
    # ==================================================
    def semantic_similarity(self, text):

        embedding = self.mpnet.encode(
            text,
            convert_to_tensor=True,
            device=DEVICE
        )

        scores = {}

        for category, cat_tensor in self.category_tensors.items():

            sim = torch.nn.functional.cosine_similarity(
                embedding.unsqueeze(0),
                cat_tensor,
                dim=1
            )

            scores[category] = float(sim.max().detach().cpu())

        # Sort categories by similarity
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        best_cat, best_score = sorted_scores[0]

        # margin = separation strength
        if len(sorted_scores) > 1:
            margin = best_score - sorted_scores[1][1]
        else:
            margin = best_score

        return best_cat, best_score, margin

    # ==================================================
    # Transformer Prediction
    # ==================================================
    def transformer_prediction(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.deberta(**inputs)

        logits = outputs.logits / self.temperature

        probs = torch.softmax(logits, dim=1)

        prob_value, pred_idx = torch.max(probs, dim=1)

        return pred_idx.item(), float(prob_value.detach().cpu())

    # ==================================================
    # Final Prediction Pipeline
    # ==================================================
    def predict(self, text):

        text = str(text)

        # ----------------------------
        # Stage 0: Rule Override
        # ----------------------------
        rule_pred = rule_override(text)

        if rule_pred:
            return rule_pred, 1.0

        # ----------------------------
        # Stage 1: Semantic Similarity
        # ----------------------------
        sem_cat, sem_score, sem_margin = self.semantic_similarity(text)

        # ----------------------------
        # Stage 2: Transformer
        # ----------------------------
        tf_pred_idx, tf_prob = self.transformer_prediction(text)

        # Load label map
        label_map_path = "models/deberta/label_map.json"

        if os.path.exists(label_map_path):
            with open(label_map_path) as f:
                label_map = json.load(f)

            reverse_map = {v: k for k, v in label_map.items()}
            tf_cat = reverse_map.get(tf_pred_idx, sem_cat)
        else:
            tf_cat = sem_cat

        # ----------------------------
        # Stage 3: Meta Model (if available)
        # ----------------------------
        if self.meta_model:

            feature_vector = np.array([[
                sem_score,
                sem_margin,
                tf_prob,
                len(text)
            ]])

            final_cat = self.meta_model.predict(feature_vector)[0]

            confidence = max(sem_score, tf_prob)

        else:
            # Fallback decision logic
            if tf_prob > 0.80:
                final_cat = tf_cat
                confidence = tf_prob
            else:
                final_cat = sem_cat
                confidence = sem_score

        return final_cat, round(float(confidence), 4)


# ==================================================
# Batch Classification Function (for dashboard.py)
# ==================================================
def classify_file(input_file, output_file):

    clf = EliteClassifier()

    import pandas as pd

    df = pd.read_excel(input_file)
    df.columns = df.columns.str.strip()

    predictions = []
    confidences = []

    for _, row in df.iterrows():
        text = " | ".join(
            [str(v) for v in row if pd.notna(v)]
        )

        pred, conf = clf.predict(text)

        predictions.append(pred)
        confidences.append(conf)

    df["Predicted Category"] = predictions
    df["Confidence"] = confidences

    df.to_excel(output_file, index=False)

    print("Classification complete.")
