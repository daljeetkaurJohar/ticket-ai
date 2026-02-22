# backend/categorization_logic.py

import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):

        self.training_df = self._load_historical_data(excel_file)

        if len(self.training_df) == 0:
            raise ValueError("No valid historical training data found.")

        print("\n✅ Training Data Loaded")
        print("Training Shape:", self.training_df.shape)
        print("\nCategory Distribution:")
        print(self.training_df["Category"].value_counts())

        self._build_vector_index()


    # ---------------------------------------------------
    # CLEAN TEXT
    # ---------------------------------------------------
    def _clean_text(self, text):

        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    # ---------------------------------------------------
    # LOAD ALL SHEETS PROPERLY
    # ---------------------------------------------------
    def _load_historical_data(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        all_data = []

        for sheet in xls.sheet_names:

            print(f"Reading sheet: {sheet}")

            df = pd.read_excel(xls, sheet)
            df.columns = df.columns.str.strip()

            if df.empty:
                continue

            if "Issue category" not in df.columns:
                continue

            if "Ticket Description" not in df.columns:
                continue

            df = df.dropna(subset=["Issue category", "Ticket Description"])

            df["text"] = df["Ticket Description"].apply(self._clean_text)

            df = df[["text", "Issue category"]]
            df = df.rename(columns={"Issue category": "Category"})

            all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        final_df = pd.concat(all_data, ignore_index=True)

        # ✅ Only allow correct 10 categories
        valid_categories = [
            "User KT issue",
            "User knowledge gap",
            "Masterdata - delayed input from user",
            "Mapping missing from user",
            "Multiple versions issue in excel",
            "Delayed logic changes from users",
            "Logic mistakes in excel vs system",
            "System Access issue",
            "System linkage issue",
            "Masterdata - incorporation in system"
        ]

        final_df = final_df[final_df["Category"].isin(valid_categories)]

        return final_df


    # ---------------------------------------------------
    # BUILD SIMILARITY MODEL
    # ---------------------------------------------------
    def _build_vector_index(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )

        self.training_vectors = self.vectorizer.fit_transform(
            self.training_df["text"]
        )

        self.training_categories = self.training_df["Category"].values


    # ---------------------------------------------------
    # PREDICT USING COSINE SIMILARITY
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

        # Threshold
        if best_score < 0.15:
            return "Needs Review", round(float(best_score), 3)

        return predicted_category, round(float(best_score), 3)
