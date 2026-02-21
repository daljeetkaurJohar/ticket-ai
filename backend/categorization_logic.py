# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):

        self.training_df = self._load_historical_data(excel_file)

        if len(self.training_df) == 0:
            raise ValueError("No training data found in issue category file.")

        self._build_vector_engine()


    def _load_historical_data(self, excel_file):

    xls = pd.ExcelFile(excel_file)
    all_data = []

    for sheet in xls.sheet_names:

        if sheet.lower() == "sheet1":
            continue

        df = pd.read_excel(xls, sheet)

        # -----------------------------
        # Detect category column safely
        # -----------------------------
        category_col = None

        for col in df.columns:
            if "category" in col.lower():
                category_col = col
                break

        if category_col is None:
            continue  # skip sheet if no category column

        # -----------------------------
        # Detect text columns safely
        # -----------------------------
        desc_col = None
        summary_col = None
        notes_col = None

        for col in df.columns:
            if "description" in col.lower():
                desc_col = col
            if "summary" in col.lower():
                summary_col = col
            if "work" in col.lower():
                notes_col = col

        df = df.dropna(subset=[category_col])

        df["combined_text"] = (
            df.get(desc_col, "").astype(str) + " " +
            df.get(summary_col, "").astype(str) + " " +
            df.get(notes_col, "").astype(str)
        )

        df["combined_text"] = df["combined_text"].apply(self._clean_text)

        df = df[[ "combined_text", category_col ]]

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
    # Clean text
    # ---------------------------------------------------
    def _clean_text(self, text):

        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    # ---------------------------------------------------
    # Build Vector Engine
    # ---------------------------------------------------
    def _build_vector_engine(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

        self.training_vectors = self.vectorizer.fit_transform(
            self.training_df["text"]
        )


    # ---------------------------------------------------
    # Categorize New Ticket
    # ---------------------------------------------------
    def categorize(self, text):

        cleaned = self._clean_text(text)

        input_vector = self.vectorizer.transform([cleaned])

        similarities = cosine_similarity(
            input_vector,
            self.training_vectors
        )

        best_index = similarities.argmax()
        best_score = similarities[0][best_index]

        predicted_category = self.training_df.iloc[best_index]["Category"]

        return predicted_category, round(float(best_score), 3)
