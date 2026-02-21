# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class CategorizationLogic:

    def __init__(self, excel_file):
        self._load_training_data(excel_file)
        self._train_model()

    # ----------------------------
    # Clean text
    # ----------------------------
    def _clean(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text.strip()

    # ----------------------------
    # Load training data
    # ----------------------------
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

            df = df.dropna(subset=["Ticket Description", "Issue category"])

            for _, row in df.iterrows():
                text = self._clean(row["Ticket Description"])
                label = str(row["Issue category"]).strip()

                if text and label and label != "nan":
                    texts.append(text)
                    labels.append(label)

        if not texts:
            raise ValueError("No valid training data found.")

        self.texts = texts
        self.labels = labels

    # ----------------------------
    # Train model
    # ----------------------------
    def _train_model(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )

        X = self.vectorizer.fit_transform(self.texts)

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        self.model.fit(X, self.labels)

    # ----------------------------
    # Predict
    # ----------------------------
    def categorize(self, text):

        text = self._clean(text)

        if not text:
            return "User Awareness", 0.0

        X_new = self.vectorizer.transform([text])

        prediction = self.model.predict(X_new)[0]
        probability = max(self.model.predict_proba(X_new)[0])

        return prediction, round(float(probability), 3)
