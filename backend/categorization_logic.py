# backend/categorization_logic.py

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):
        self.rules_df = self._load_all_rows(excel_file)
        self._build_vector_engine()
        self._build_issue_counts()

    # --------------------------------
    # Clean Text
    # --------------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # --------------------------------
    # Load ALL rows
    # --------------------------------
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

    # --------------------------------
    # Build TF-IDF Engine
    # --------------------------------
    def _build_vector_engine(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.row_vectors = self.vectorizer.fit_transform(
            self.rules_df["text"]
        )

    # --------------------------------
    # Count rows per issue (for normalization)
    # --------------------------------
    def _build_issue_counts(self):
        self.issue_counts = self.rules_df["issue"].value_counts().to_dict()

    # --------------------------------
    # FINAL Categorize
    # --------------------------------
    def categorize(self, text):

        text = self._clean(text)

        text_vector = self.vectorizer.transform([text])

        similarities = cosine_similarity(
            text_vector,
            self.row_vectors
        )[0]

        # Get Top-K similar rows
        TOP_K = 7
        top_indices = similarities.argsort()[-TOP_K:]

        issue_scores = {}

        for idx in top_indices:

            issue = self.rules_df.iloc[idx]["issue"]
            score = similarities[idx]

            if issue not in issue_scores:
                issue_scores[issue] = 0

            issue_scores[issue] += score

        # Normalize by issue size
        for issue in issue_scores:
            issue_scores[issue] /= self.issue_counts.get(issue, 1)

        # Optional: penalize generic category
        if "User knowledge gap" in issue_scores:
            issue_scores["User knowledge gap"] *= 0.85

        if not issue_scores:
            return "Needs Manual Review", 0.0

        best_issue = max(issue_scores, key=issue_scores.get)
        best_score = issue_scores[best_issue]

        if best_score < 0.03:
            return "Needs Manual Review", round(float(best_score), 3)

        return best_issue, round(float(best_score), 3)
