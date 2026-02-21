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

        self._build_vector_engine()


    # ---------------------------------------------------
    # CLEAN TEXT
    # ---------------------------------------------------
    def _clean_text(self, text):

        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    # ---------------------------------------------------
    # LOAD HISTORICAL DATA FROM ALL SHEETS
    # ---------------------------------------------------
    def _load_historical_data(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        all_data = []

        for sheet in xls.sheet_names:

            # Skip unwanted sheet if needed
            if sheet.lower() == "sheet1":
                continue

            df = pd.read_excel(xls, sheet)

            if df.empty:
                continue

            # ------------------------------
            # Detect Category Column
            # ------------------------------
            category_col = None
            for col in df.columns:
                if "category" in col.lower():
                    category_col = col
                    break

            if category_col is None:
                continue

            # ------------------------------
            # Detect Text Columns
            # ------------------------------
            desc_col = None
            summary_col = None
            notes_col = None

            for col in df.columns:
                col_lower = col.lower()

                if "description" in col_lower:
                    desc_col = col

                if "summary" in col_lower:
                    summary_col = col

                if "work" in col_lower:
                    notes_col = col

            df = df.dropna(subset=[category_col])

            if df.empty:
                continue

            # ------------------------------
            # Safe Text Combination
            # ------------------------------
            desc_series = df[desc_col].astype(str) if desc_col and desc_col in df.columns else ""
            summary_series = df[summary_col].astype(str) if summary_col and summary_col in df.columns else ""
            notes_series = df[notes_col].astype(str) if notes_col and notes_col in df.columns else ""

            df["combined_text"] = (
                desc_series + " " +
                summary_series + " " +
                notes_series
            )

            df["combined_text"] = df["combined_text"].apply(self._clean_text)

            df = df[["combined_text", category_col]]

            df = df.rename(columns={
                "combined_text": "text",
                category_col: "Category"
            })

            all_data.append(df)

        if len(all_data) == 0:
            return pd.DataFrame()

        final_df = pd.concat(all_data, ignore_index=True)

        return final_df


    # ---------------------------------------------------
    # BUILD VECTOR ENGINE
    # ---------------------------------------------------
    def _build_vector_engine(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1  # IMPORTANT FIX
        )

        self.training_vectors = self.vectorizer.fit_transform(
            self.training_df["text"]
        )


    # ---------------------------------------------------
    # CATEGORIZE NEW TICKET (FIXED VERSION)
    # ---------------------------------------------------
    def categorize(self, text):

        cleaned = self._clean_text(text)

        input_vector = self.vectorizer.transform([cleaned])

        similarities = cosine_similarity(
            input_vector,
            self.training_vectors
        )[0]

        # Add similarity scores to dataframe
        temp_df = self.training_df.copy()
        temp_df["similarity"] = similarities

        # ---- CATEGORY LEVEL AGGREGATION (MAIN FIX) ----
        category_scores = (
            temp_df.groupby("Category")["similarity"]
            .mean()
            .sort_values(ascending=False)
        )

        predicted_category = category_scores.index[0]
        confidence = category_scores.iloc[0]

        return predicted_category, round(float(confidence), 3)
