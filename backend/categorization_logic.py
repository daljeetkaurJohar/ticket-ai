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

            # Skip Sheet1 (rule definition sheet)
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
    # Build TF-IDF engine
    # ---------------------------------
    def _build_vector_engine(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )

        self.training_vectors = self.vectorizer.fit_transform(
            self.training_df["text"]
        )

    # ---------------------------------
    # Categorize using Top-K Voting
    # ---------------------------------
    def categorize(self, text):

        text = self._clean(text)

        text_vector = self.vectorizer.transform([text])

        similarities = cosine_similarity(
            text_vector,
            self.training_vectors
        )[0]

        # Take top 10 similar tickets
        TOP_K = 10
        top_indices = similarities.argsort()[-TOP_K:]

        issue_scores = {}

        for idx in top_indices:

            issue = self.training_df.iloc[idx]["issue"]
            score = similarities[idx]

            if issue not in issue_scores:
                issue_scores[issue] = 0

            issue_scores[issue] += score

        # Normalize by category size (reduces dominance)
        for issue in issue_scores:
            category_count = len(
                self.training_df[self.training_df["issue"] == issue]
            )
            issue_scores[issue] /= category_count

        # Select best category
        best_issue = max(issue_scores, key=issue_scores.get)
        best_score = issue_scores[best_issue]

        # Weak similarity fallback
        if best_score < 0.05:
            return "User Awareness", round(float(best_score), 3)

        return best_issue, round(float(best_score), 3)
