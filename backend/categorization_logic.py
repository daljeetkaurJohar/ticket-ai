# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):

        self.training_df = self._load_historical_data(excel_file)

        if len(self.training_df) == 0:
            raise ValueError("No valid historical training data found.")

        print("\n✅ Training Data Loaded Successfully")
        print("Training Shape:", self.training_df.shape)
        print("Category Distribution:")
        print(self.training_df["Category"].value_counts())

        self._build_model()

    # ---------------------------------------------------
    # CLEAN TEXT
    # ---------------------------------------------------
    def _clean_text(self, text):

        text = str(text).lower()

        # Remove timestamps
        text = re.sub(r"\d{2}-\d{2}-\d{4}.*?(am|pm)", "", text)

        # Remove special characters
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ---------------------------------------------------
    # LOAD TRAINING DATA FROM ALL SHEETS
    # ---------------------------------------------------
    def _load_historical_data(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        all_data = []

        for sheet in xls.sheet_names:

            print(f"Reading sheet: {sheet}")

            df = pd.read_excel(xls, sheet)

            if df.empty:
                print(f"Skipping {sheet} — empty sheet")
                continue

            if "Issue" not in df.columns:
                print(f"Skipping {sheet} — no 'Issue' column found")
                continue

            df = df.dropna(subset=["Issue"])

            if df.empty:
                continue

            # Combine all non-Issue columns as training text
            text_columns = [col for col in df.columns if col != "Issue"]

            df["text"] = df[text_columns].astype(str).agg(" ".join, axis=1)
            df["text"] = df["text"].apply(self._clean_text)

            df = df[["text", "Issue"]]
            df = df.rename(columns={"Issue": "Category"})

            all_data.append(df)

        if len(all_data) == 0:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    # ---------------------------------------------------
    # BUILD SIMILARITY MODEL
    # ---------------------------------------------------
    def _build_model(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )

        # Fit vectorizer
        self.training_vectors = self.vectorizer.fit_transform(
            self.training_df["text"]
        )

        self.training_categories = self.training_df["Category"].values

        print("✅ Similarity model built successfully")

    # ---------------------------------------------------
    # PREDICT CATEGORY USING COSINE SIMILARITY
    # ---------------------------------------------------
    def categorize(self, text):
    
        cleaned = self._clean_text(text)
    
        if not cleaned:
            return "Insufficient Data", 0.0
    
        new_vector = self.vectorizer.transform([cleaned])
    
        similarities = cosine_similarity(
            new_vector,
            self.training_vectors
        )[0]
    
        best_index = similarities.argmax()
        best_score = similarities[best_index]
    
        predicted_category = self.training_categories[best_index]
    
        # Optional threshold
        if best_score < 0.20:
            return "Needs Review", round(float(best_score), 3)
    
        return predicted_category, round(float(best_score), 3)
