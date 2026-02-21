# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class CategorizationLogic:

    def __init__(self, excel_file):

        self.training_df = self._load_historical_data(excel_file)

        if len(self.training_df) == 0:
            raise ValueError("No valid historical training data found.")

        print("\nTraining Data Loaded")
        print(self.training_df["Category"].value_counts())

        self._build_model()


    # ---------------------------------------------------
    # CLEAN TEXT
    # ---------------------------------------------------
    def _clean_text(self, text):

        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    # ---------------------------------------------------
    # LOAD TRAINING DATA
    # ---------------------------------------------------
    def _load_historical_data(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        all_data = []

        for sheet in xls.sheet_names:

            df = pd.read_excel(xls, sheet)

            if df.empty:
                continue

            if "Issue" not in df.columns:
                continue

            category_col = "Issue"

            desc_col = "Description" if "Description" in df.columns else None
            issue_desc_col = "Issue Description" if "Issue Description" in df.columns else None

            df = df.dropna(subset=[category_col])

            if df.empty:
                continue

            desc_series = df[desc_col].astype(str) if desc_col else ""
            issue_desc_series = df[issue_desc_col].astype(str) if issue_desc_col else ""

            df["text"] = (desc_series + " " + issue_desc_series).apply(self._clean_text)

            df = df[["text", category_col]]
            df = df.rename(columns={category_col: "Category"})

            all_data.append(df)

        if len(all_data) == 0:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)


    # ---------------------------------------------------
    # BUILD ML MODEL
    # ---------------------------------------------------
    def _build_model(self):

        self.label_encoder = LabelEncoder()

        y_encoded = self.label_encoder.fit_transform(
            self.training_df["Category"]
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1
            )),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            ))
        ])

        self.pipeline.fit(self.training_df["text"], y_encoded)


    # ---------------------------------------------------
    # PREDICT
    # ---------------------------------------------------
    def categorize(self, text):

        cleaned = self._clean_text(text)

        pred_encoded = self.pipeline.predict([cleaned])[0]
        probabilities = self.pipeline.predict_proba([cleaned])[0]

        confidence = max(probabilities)

        predicted_category = self.label_encoder.inverse_transform([pred_encoded])[0]

        return predicted_category, round(float(confidence), 3)
