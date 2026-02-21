# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):
        self.training_df = self._load_historical_data(excel_file)
        self._build_vector_engine()

    # ---------------------------------
    # Clean text
    # ---------------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # ---------------------------------
    # Load historical sheets (NOT Sheet1)
    # ---------------------------------
    def _load_historical_data(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        sheets = xls.sheet_names

        rows = []

        for sheet in sheets:

            if sheet.lower() == "sheet1":
                continue

            df = pd.read_excel(xls, sheet)

            if "Issue" not in df.columns:
                continue

            for _, row in df.iterrows():

                issue = str(row.get("Issue", "")).strip()

                # Combine ALL columns dynamically
                text = " ".join([
                    str(row[col]) for col in df.columns
                ])

                text = self._clean(text)

                if issue and text.strip():
                    rows.append({
                        "issue": issue,
                        "text": text
                    })

        if not rows:
            raise ValueError("No historical training data loaded.")

        return pd.DataFrame(rows)

    # ---------------------------------
    # Build TF-IDF vector engine
    # ---------------------------------
    def _build_vector_engine(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.training_vectors = self.vectorizer.fit_transform(
            self.training_df["text"]
        )

    # ---------------------------------
    # Categorize using nearest similarity
    # ---------------------------------
    def categorize(self, text):

        text = self._clean(text)

        text_vector = self.vectorizer.transform([text])

        similarities = cosine_similarity(
            text_vector,
            self.training_vectors
        )[0]

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        if best_score < 0.1:
            return "User Awareness", round(float(best_score), 3)

        best_issue = self.training_df.iloc[best_index]["issue"]

        return best_issue, round(float(best_score), 3)
