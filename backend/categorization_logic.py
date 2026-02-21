# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):
        self.issue_corpus = self._build_category_corpus(excel_file)
        self._build_vector_model()

    # ---------------------------------
    # Clean text
    # ---------------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # ---------------------------------
    # Club all ticket descriptions per category
    # ---------------------------------
    def _build_category_corpus(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        sheets = xls.sheet_names

        category_text = {}

        for sheet in sheets:

            if sheet.lower() == "sheet1":
                continue

            df = pd.read_excel(xls, sheet)

            if "Ticket Description" not in df.columns:
                continue

            if "Issue category" not in df.columns:
                continue

            for _, row in df.iterrows():

                category = str(row["Issue category"]).strip()
                text = self._clean(row["Ticket Description"])

                if category not in category_text:
                    category_text[category] = ""

                category_text[category] += " " + text

        if not category_text:
            raise ValueError("No historical ticket data found.")

        return category_text

    # ---------------------------------
    # Build TF-IDF vectors per category
    # ---------------------------------
    def _build_vector_model(self):

        self.categories = list(self.issue_corpus.keys())
        corpus = [self.issue_corpus[cat] for cat in self.categories]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

        self.category_vectors = self.vectorizer.fit_transform(corpus)

    # ---------------------------------
    # Categorize new ticket
    # ---------------------------------
    def categorize(self, text):

        text = self._clean(text)

        new_vector = self.vectorizer.transform([text])

        similarities = cosine_similarity(
            new_vector,
            self.category_vectors
        )[0]

        best_index = similarities.argmax()
        best_score = similarities[best_index]
        best_category = self.categories[best_index]

        return best_category, round(float(best_score), 3)
