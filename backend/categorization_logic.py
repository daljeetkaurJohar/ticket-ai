# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    def __init__(self, excel_file):

        self.training_df = self._load_historical_data(excel_file)

        if len(self.training_df) == 0:
            raise ValueError("No valid historical training data found.")

        print("\n✅ Training Data Loaded Successfully")
        print("Training Shape:", self.training_df.shape)
        print("Category Distribution:")
        print(self.training_df["Category"].value_counts())

        self._build_vector_engine()


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

            print(f"Reading sheet: {sheet}")

            df = pd.read_excel(xls, sheet)

            if df.empty:
                print(f"Skipping {sheet} — empty sheet")
                continue

            # ------------------------------
            # FORCE CATEGORY COLUMN = "Issue"
            # ------------------------------
            if "Issue" not in df.columns:
                print(f"Skipping {sheet} — no 'Issue' column found")
                continue

            category_col = "Issue"

            # ------------------------------
            # TEXT COLUMNS
            # ------------------------------
            desc_col = None
            issue_desc_col = None

            for col in df.columns:
                if col.lower() == "description":
                    desc_col = col

                if "issue description" in col.lower():
                    issue_desc_col = col

            if desc_col is None and issue_desc_col is None:
                print(f"Skipping {sheet} — no description columns found")
                continue

            df = df.dropna(subset=[category_col])

            if df.empty:
                continue

            desc_series = df[desc_col].astype(str) if desc_col else ""
            issue_desc_series = df[issue_desc_col].astype(str) if issue_desc_col else ""

            df["combined_text"] = desc_series + " " + issue_desc_series
            df["combined_text"] = df["combined_text"].apply(self._clean_text)

            df = df[["combined_text", category_col]]

            df = df.rename(columns={
                "combined_text": "text",
                category_col: "Category"
            })

            all_data.append(df)

        if len(all_data) == 0:
            return pd.DataFrame()

        final_df = pd.concat(all_data, ignore_index=True)

        return final_df


    # ---------------------------------------------------
    # BUILD TF-IDF ENGINE
    # ---------------------------------------------------
    def _build_vector_engine(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )

        self.training_vectors = self.vectorizer.fit_transform(
            self.training_df["text"]
        )


    # ---------------------------------------------------
    # CATEGORIZE NEW TICKET
    # ---------------------------------------------------
    def categorize(self, text):

        cleaned = self._clean_text(text)

        input_vector = self.vectorizer.transform([cleaned])

        similarities = cosine_similarity(
            input_vector,
            self.training_vectors
        )[0]

        temp_df = self.training_df.copy()
        temp_df["similarity"] = similarities

        # CATEGORY LEVEL AGGREGATION
        category_scores = (
            temp_df.groupby("Category")["similarity"]
            .mean()
            .sort_values(ascending=False)
        )

        predicted_category = category_scores.index[0]
        confidence = category_scores.iloc[0]

        return predicted_category, round(float(confidence), 3)
