# backend/categorization_logic.py

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CategorizationLogic:

    # ----------------------------------------
    # LOCKED ENTERPRISE TAXONOMY (ONLY 10)
    # ----------------------------------------
    ALLOWED_CATEGORIES = [
        "User KT issue",
        "User knowledge gap",
        "Masterdata - delayed input from user",
        "Mapping missing from user",
        "Multiple versions issue in excel",
        "Delayed logic changes from users",
        "Logic mistakes in excel vs system",
        "System Access issue",
        "System linkage issue",
        "Masterdata - incorporation in system"
    ]

    # ----------------------------------------
    # INIT
    # ----------------------------------------
    def __init__(self, excel_file):

        self.training_df = self._load_historical_data(excel_file)

        if self.training_df.empty:
            raise ValueError("No valid training data found after normalization.")

        print("\nTraining Size:", self.training_df.shape)
        print("\nFinal Category Distribution:")
        print(self.training_df["Category"].value_counts())

        self._build_vector_index()

    # ----------------------------------------
    # CLEAN TEXT
    # ----------------------------------------
    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ----------------------------------------
    # LOAD & NORMALIZE TRAINING DATA
    # ----------------------------------------
    def _load_historical_data(self, excel_file):

        xls = pd.ExcelFile(excel_file)
        all_data = []

        # RAW → STANDARD CATEGORY MAPPING
        category_mapping = {

            "User Awareness": "User knowledge gap",
            "User : Business Logic Issue": "Logic mistakes in excel vs system",
            "IT : Access": "System Access issue",
            "IT : System Issue": "System linkage issue",
            "User : Mappings Missing": "Mapping missing from user",
            "User : Master Data": "Masterdata - delayed input from user",
            "USer : Master Data": "Masterdata - delayed input from user",
            "Master Data Issue": "Masterdata - incorporation in system",
        }

        for sheet in xls.sheet_names:

            df = pd.read_excel(xls, sheet)
            df.columns = df.columns.str.strip()

            if "Issue category" not in df.columns:
                continue

            if "Ticket Description" not in df.columns:
                continue

            df = df.dropna(subset=["Issue category", "Ticket Description"])

            # NORMALIZE LABELS
            df["Issue category"] = df["Issue category"].replace(category_mapping)

            # KEEP ONLY APPROVED 10
            df = df[df["Issue category"].isin(self.ALLOWED_CATEGORIES)]

            if df.empty:
                continue

            df["text"] = df["Ticket Description"].apply(self._clean_text)

            df = df[["text", "Issue category"]]
            df = df.rename(columns={"Issue category": "Category"})

            all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    # ----------------------------------------
    # BUILD TF-IDF INDEX
    # ----------------------------------------
    def _build_vector_index(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )

        self.training_vectors = self.vectorizer.fit_transform(
            self.training_df["text"]
        )

        self.training_categories = self.training_df["Category"].values

    # ----------------------------------------
    # HYBRID CATEGORIZATION
    # ----------------------------------------
    def categorize(self, text):

        cleaned = self._clean_text(text)

        if not cleaned:
            return "Needs Review", 0.0

        # ---------- RULE LAYER ----------

        if any(k in cleaned for k in ["access", "authorization", "login", "permission"]):
            return "System Access issue", 0.95

        if any(k in cleaned for k in ["linkage", "integration", "interface", "sync"]):
            return "System linkage issue", 0.95

        if any(k in cleaned for k in ["mapping", "mapped"]):
            return "Mapping missing from user", 0.95

        if any(k in cleaned for k in ["master data", "masterdata", "new product"]):
            return "Masterdata - incorporation in system", 0.90

        if any(k in cleaned for k in ["awaiting user", "pending from user"]):
            return "Masterdata - delayed input from user", 0.90

        if any(k in cleaned for k in ["formula", "logic error", "calculation"]):
            return "Logic mistakes in excel vs system", 0.90

        if any(k in cleaned for k in ["version mismatch", "multiple version"]):
            return "Multiple versions issue in excel", 0.90

        if any(k in cleaned for k in ["kt", "knowledge transfer", "training"]):
            return "User KT issue", 0.85

        if any(k in cleaned for k in ["guidance", "clarification", "not aware"]):
            return "User knowledge gap", 0.85

        # ---------- ML FALLBACK ----------

        new_vector = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(new_vector, self.training_vectors)[0]

        best_index = similarities.argmax()
        best_score = similarities[best_index]
        predicted_category = self.training_categories[best_index]

        if best_score < 0.25:
            return "Needs Review", round(float(best_score), 3)

        return predicted_category, round(float(best_score), 3)
