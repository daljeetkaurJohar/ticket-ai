# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class CategorizationLogic:

    def __init__(self, excel_file):
        self._load_training_data(excel_file)
        self._train_model()

    # ---------------------------------
    # Clean text
    # ---------------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # ---------------------------------
    # Load historical training data
    # ---------------------------------
    def _load_training_data(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        sheets = xls.sheet_names

        texts = []
        labels = []

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
                label = str(row["Issue category"]).strip()

                if text and label:
                    texts.append(text)
                    labels.append(label)

        self.texts = texts
        self.labels = labels

    # ---------------------------------
    # Train classifier
    # ---------------------------------
    def _train_model(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

        X = self.vectorizer.fit_transform(self.texts)

        # Balanced class weight prevents User Awareness dominance
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        self.model.fit(X, self.labels)

    # ---------------------------------
    # Categorize new ticket
    # ---------------------------------
    def categorize(self, text):

        text = self._clean(text)

        X_new = self.vectorizer.transform([text])

        prediction = self.model.predict(X_new)[0]
        prob = max(self.model.predict_proba(X_new)[0])

        return prediction, round(float(prob), 3)
