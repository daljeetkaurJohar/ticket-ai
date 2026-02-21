# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):
        self.issue_text_map = self._build_issue_text_map(excel_file)
        self._build_vector_engine()

    # -----------------------------
    # Clean text
    # -----------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # -----------------------------
    # Build issue cluster texts
    # -----------------------------
    def _build_issue_text_map(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        sheets = xls.sheet_names

        issue_map = {}

        for sheet in sheets:
            df = pd.read_excel(xls, sheet)

            if "Issue" not in df.columns:
                continue

            for _, row in df.iterrows():

                issue = str(row.get("Issue", "")).strip()

                combined_text = " ".join([
                    str(row.get("Description", "")),
                    str(row.get("Issue Description", "")),
                    str(row.get("Remarks", "")),
                    str(row.get("Unnamed: 5", ""))
                ])

                combined_text = self._clean(combined_text)

                if not issue:
                    continue

                if issue not in issue_map:
                    issue_map[issue] = ""

                issue_map[issue] += " " + combined_text

        return issue_map

    # -----------------------------
    # Build TF-IDF engine
    # -----------------------------
    def _build_vector_engine(self):

        self.issues = list(self.issue_text_map.keys())
        corpus = [self.issue_text_map[i] for i in self.issues]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )

        self.issue_vectors = self.vectorizer.fit_transform(corpus)

    # -----------------------------
    # Categorize
    # -----------------------------
    def categorize(self, text):

        text = self._clean(text)

        text_vector = self.vectorizer.transform([text])

        similarities = cosine_similarity(text_vector, self.issue_vectors)[0]

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        if best_score < 0.05:
            return "Needs Manual Review", round(float(best_score), 3)

        return self.issues[best_index], round(float(best_score), 3)
