# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):

        self.sheet1_df = self._load_sheet1(excel_file)
        self.history_df = self._load_historical(excel_file)

        self._build_vector_models()

    # ---------------------------------
    # Clean text
    # ---------------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # ---------------------------------
    # Load Sheet1 (issue definitions)
    # ---------------------------------
    def _load_sheet1(self, excel_file):

        df = pd.read_excel(excel_file, sheet_name="Sheet1")

        rows = []

        for _, row in df.iterrows():
            issue = str(row["Issue"]).strip()

            text = " ".join([
                str(row.get("Description", "")),
                str(row.get("Issue Description", ""))
            ])

            text = self._clean(text)

            if issue and text:
                rows.append({
                    "issue": issue,
                    "text": text
                })

        return pd.DataFrame(rows)

    # ---------------------------------
    # Load historical sheets
    # ---------------------------------
    def _load_historical(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        sheets = xls.sheet_names

        rows = []

        for sheet in sheets:

            if sheet.lower() == "sheet1":
                continue

            df = pd.read_excel(xls, sheet)

            if "Ticket Description" not in df.columns:
                continue

            if "Issue category" not in df.columns:
                continue

            for _, row in df.iterrows():

                text = self._clean(row["Ticket Description"])
                issue = str(row["Issue category"]).strip()

                if issue and text:
                    rows.append({
                        "issue": issue,
                        "text": text
                    })

        return pd.DataFrame(rows)

    # ---------------------------------
    # Build vector models
    # ---------------------------------
    def _build_vector_models(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

        combined_corpus = list(self.sheet1_df["text"]) + list(self.history_df["text"])

        self.vectorizer.fit(combined_corpus)

        self.sheet1_vectors = self.vectorizer.transform(self.sheet1_df["text"])
        self.history_vectors = self.vectorizer.transform(self.history_df["text"])

    # ---------------------------------
    # Categorize ticket
    # ---------------------------------
    def categorize(self, text):

        text = self._clean(text)
        new_vector = self.vectorizer.transform([text])

        # Similarity with Sheet1
        sim_sheet1 = cosine_similarity(new_vector, self.sheet1_vectors)[0]
        best_sheet1_idx = sim_sheet1.argmax()
        best_sheet1_score = sim_sheet1[best_sheet1_idx]
        best_sheet1_issue = self.sheet1_df.iloc[best_sheet1_idx]["issue"]

        # Similarity with historical
        sim_history = cosine_similarity(new_vector, self.history_vectors)[0]
        best_history_idx = sim_history.argmax()
        best_history_score = sim_history[best_history_idx]
        best_history_issue = self.history_df.iloc[best_history_idx]["issue"]

        # Decide final category
        if best_sheet1_score > best_history_score:
            return best_sheet1_issue, round(float(best_sheet1_score), 3)
        else:
            return best_history_issue, round(float(best_history_score), 3)
