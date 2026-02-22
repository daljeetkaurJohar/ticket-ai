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

        print("\n✅ Training Data Loaded")
        print("Training Shape:", self.training_df.shape)
        print("\nCategory Distribution (After Cleaning):")
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
    # LOAD + NORMALIZE DATA
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

            # ---------------------------------------------------
            # CATEGORY NORMALIZATION MAP
            # ---------------------------------------------------
            category_mapping = {

                # User-related messy labels
                "User Awareness": "User knowledge gap",
                "User awareness": "User knowledge gap",
                "User : Business Logic Issue": "Logic mistakes in excel vs system",
                "User : Mappings Missing": "Mapping missing from user",
                "User : Master Data": "Masterdata - delayed input from user",
                "USer : Master Data": "Masterdata - delayed input from user",

                # IT-related messy labels
                "IT : Access": "System Access issue",
                "IT : System Issue": "System linkage issue",
                "IT : Change": "System linkage issue",
                "IT : Master Data": "Masterdata - incorporation in system",
                "Master Data Issue": "Masterdata - incorporation in system",
            }

            df["Issue category"] = df["Issue category"].replace(category_mapping)

            # ---------------------------------------------------
            # KEEP ONLY VALID FINAL CATEGORIES
            # ---------------------------------------------------
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

            df = df[df["Issue category"].isin(valid_categories)]

            # Clean text
            df["text"] = df["Ticket Description"].apply(self._clean_text)

            df = df[["text", "Issue category"]]
            df = df.rename(columns={"Issue category": "Category"})

            all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        final_df = pd.concat(all_data, ignore_index=True)

        return final_df


    # ---------------------------------------------------
    # BUILD VECTOR INDEX
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
    # PREDICT USING SIMILARITY
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

        # Confidence threshold
        if best_score < 0.18:
            return "Needs Review", round(float(best_score), 3)

        return predicted_category, round(float(best_score), 3)
