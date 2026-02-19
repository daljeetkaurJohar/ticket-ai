import pandas as pd
import numpy as np
import json
import os
import torch
from sentence_transformers import SentenceTransformer, util

# Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
JSON_FILE = "data/category_examples.json"
AUTO_LEARN_FILE = "data/auto_learn.xlsx"
CONFIDENCE_THRESHOLD = 0.85

class EnterpriseIntentClassifier:
    def __init__(self):
        print(f"Initializing {MODEL_NAME}...")
        
        if not os.path.exists(JSON_FILE):
            raise FileNotFoundError(f"Missing essential file: {JSON_FILE}")

        # Load model and move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)

        # Load and flatten the nested JSON structure
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        all_categories_data = {}
        for group_name, categories in raw_data.items():
            all_categories_data.update(categories)

        self.categories = []
        centroid_list = []

        print("Building category centroids...")
        for category, content in all_categories_data.items():
            text_blobs = []
            
            # Extract strings from JSON keys
            if "examples" in content: text_blobs.extend(content["examples"])
            if "symptoms" in content: text_blobs.extend(content["symptoms"])
            if "causes" in content: text_blobs.extend(content["causes"])
            if "definition" in content: text_blobs.append(content["definition"])

            if text_blobs:
                # Generate embeddings for all text related to this category
                vectors = self.model.encode(text_blobs, convert_to_numpy=True)
                # Average them to create a single 'concept' vector (centroid)
                centroid = np.mean(vectors, axis=0)
                centroid_list.append(centroid)
                self.categories.append(category)

        # CRITICAL: Convert centroids to a Torch Tensor for compatibility with util.cos_sim
        self.centroids = torch.tensor(np.vstack(centroid_list)).to(self.device)
        
        print(f"Classifier ready with {len(self.categories)} categories.")

    def build_context(self, row):
        """Combines specific columns with weights for semantic searching."""
        weights = {
            "Ticket Summary": 2,
            "Ticket Details": 3,
            "Solution": 3,
            "Work notes": 2
        }
        parts = []
        for col, weight in weights.items():
            if col in row and pd.notna(row[col]):
                # Multiply text presence to emphasize important columns
                parts.extend([str(row[col])] * weight)
        return " | ".join(parts)

    def predict_batch(self, df):
        """High-performance batch prediction using matrix multiplication."""
        if df.empty:
            return [], []

        # 1. Prepare text data
        contexts = df.apply(self.build_context, axis=1).tolist()

        # 2. Encode all rows at once
        embeddings = self.model.encode(
            contexts,
            batch_size=32,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # 3. Vectorized Similarity (Calculates similarity of every ticket vs every category)
        # Returns a matrix of shape [num_tickets, num_categories]
        sim_matrix = util.cos_sim(embeddings, self.centroids)

        # 4. Extract best matches and scores
        best_scores, best_indices = torch.max(sim_matrix, dim=1)

        # Convert back to labels and standard python types
        predicted_labels = [self.categories[idx] for idx in best_indices.cpu().numpy()]
        confidence_scores = best_scores.cpu().numpy().tolist()

        return predicted_labels, confidence_scores

    def auto_learn(self, df):
        """Filters high-confidence predictions for future training."""
        high_conf = df[df["Confidence"] >= CONFIDENCE_THRESHOLD].copy()
        
        if high_conf.empty:
            return

        if os.path.exists(AUTO_LEARN_FILE):
            try:
                old_df = pd.read_excel(AUTO_LEARN_FILE)
                combined = pd.concat([old_df, high_conf], ignore_index=True)
            except Exception:
                combined = high_conf
        else:
            combined = high_conf

        combined.to_excel(AUTO_LEARN_FILE, index=False)
        print(f"Auto-learn: Captured {len(high_conf)} new high-confidence examples.")

def classify_file(input_path, output_path):
    """Main entry point for file processing."""
    clf = EnterpriseIntentClassifier()
    
    # Load data
    df = pd.read_excel(input_path)
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Predict
    pred, conf = clf.predict_batch(df)
    
    # Append results
    df["Predicted Category"] = pred
    df["Confidence"] = conf

    # Run auto-learning
    clf.auto_learn(df)

    # Save results
    df.to_excel(output_path, index=False)
    print(f"Success: Processed file saved to {output_path}")

# Example usage:
# if __name__ == "__main__":
#     classify_file("input.xlsx", "output.xlsx")
