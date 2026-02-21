# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):
        self.rules_df = self._load_all_rows(excel_file)
        self._build_vector_engine()

    # -----------------------------
    # Clean text
    # -----------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # -----------------------------
    # Load ALL rows from ALL sheets
    # -----------------------------
    def _load_all_rows(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        rows = []

        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet)

            if "Issue" not in df.columns:
                continue

            for _, row in df.iterrows():

                issue = str(row.get("Issue", "")).strip()

                text = " ".join([
                    str(row.get("Description", "")),
                    str(row.get("Issue Description", "")),
                    str(row.get("Remarks", "")),
                    str(row.get("Unnamed: 5", ""))
                ])

                text = self._clean(text)

                if issue and text.strip():
                    rows.append({
                        "issue": issue,
                        "text": text
                    })

        return pd.DataFrame(rows)

    # -----------------------------
    # Build TF-IDF on EACH ROW
    # -----------------------------
    def _build_vector_engine(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )

        self.row_vectors = self.vectorizer.fit_transform(
            self.rules_df["text"]
        )

    # -----------------------------
    # Categorize
    # -----------------------------
    def categorize(self, text):

        text = self._clean(text)

        text_vector = self.vectorizer.transform([text])

        similarities = cosine_similarity(
            text_vector,
            self.row_vectors
        )[0]

        best_row_index = similarities.argmax()
        best_score = similarities[best_row_index]

        if best_score < 0.05:
            return "Needs Manual Review", round(float(best_score), 3)

        best_issue = self.rules_df.iloc[best_row_index]["issue"]

        return best_issue, round(float(best_score), 3)
