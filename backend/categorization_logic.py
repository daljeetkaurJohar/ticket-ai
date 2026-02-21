# backend/categorization_logic.py

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):
        self.issue_texts = self._load_historical_data(excel_file)
        self._build_centroid_model()

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

        issue_text_map = {}

        for sheet in sheets:

            if sheet.lower() == "sheet1":
                continue

            df = pd.read_excel(xls, sheet)

            if "Issue" not in df.columns:
                continue

            for _, row in df.iterrows():

                issue = str(row.get("Issue", "")).strip()

                text = " ".join(
                    str(row[col]) for col in df.columns
                )

                text = self._clean(text)

                if issue and text.strip():

                    if issue not in issue_text_map:
                        issue_text_map[issue] = []

                    issue_text_map[issue].append(text)

        if not issue_text_map:
            raise ValueError("No historical data found.")

        return issue_text_map

    # ---------------------------------
    # Build centroid model
    # ---------------------------------
    def _build_centroid_model(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

        # Flatten all texts
        all_texts = []
        issue_labels = []

        for issue, texts in self.issue_texts.items():
            for t in texts:
                all_texts.append(t)
                issue_labels.append(issue)

        vectors = self.vectorizer.fit_transform(all_texts)

        # Compute centroid per issue
        self.issue_centroids = {}
        self.issues = list(self.issue_texts.keys())

        for issue in self.issues:
            indices = [
                i for i, label in enumerate(issue_labels)
                if label == issue
            ]
            centroid = np.mean(vectors[indices].toarray(), axis=0)
            self.issue_centroids[issue] = centroid

    # ---------------------------------
    # Categorize
    # ---------------------------------
    def categorize(self, text):

        text = self._clean(text)

        text_vector = self.vectorizer.transform([text]).toarray()[0]

        best_issue = None
        best_score = 0

        for issue, centroid in self.issue_centroids.items():

            similarity = cosine_similarity(
                [text_vector],
                [centroid]
            )[0][0]

            if similarity > best_score:
                best_score = similarity
                best_issue = issue

        if best_score < 0.05:
            return "User Awareness", round(float(best_score), 3)

        return best_issue, round(float(best_score), 3)
