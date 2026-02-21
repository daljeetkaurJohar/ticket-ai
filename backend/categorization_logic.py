# backend/categorization_logic.py

import pandas as pd
import re
from collections import defaultdict


class CategorizationLogic:

    def __init__(self, excel_file):
        self.category_keywords = self._build_keyword_dictionary(excel_file)

    # ----------------------------------
    # Clean Text
    # ----------------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # ----------------------------------
    # Extract Keywords Per Category
    # ----------------------------------
    def _build_keyword_dictionary(self, excel_file):

        df = pd.read_excel(excel_file, sheet_name=0)

        keyword_dict = defaultdict(set)

        for _, row in df.iterrows():

            raw_issue = str(row.get("Issue", "")).strip()

            combined_text = " ".join([
                str(row.get("Description", "")),
                str(row.get("Issue Description", "")),
                str(row.get("Unnamed: 5", ""))
            ])

            combined_text = self._clean(combined_text)

            words = [
                w for w in combined_text.split()
                if len(w) > 3
            ]

            final_category = self._map_category(raw_issue)

            keyword_dict[final_category].update(words)

        return keyword_dict

    # ----------------------------------
    # Map Raw Excel Issue → Final 8 Categories
    # ----------------------------------
    def _map_category(self, raw_issue):

        mapping = {
            "System linkage issue": "IT : System Issue",
            "System Access issue": "IT : Access",
            "IT Masterdata issue": "IT : Master Data",
            "Mapping missing from user": "User : Mappings Missing",
            "Masterdata - delayed input from user": "User : Master Data",
            "Logic mistakes in excel vs system": "User : Business Logic Issue",
            "Multiple versions issue in excel": "User : Business Logic Issue",
            "User knowledge gap": "User Awareness",
            "User KT issue": "User Awareness",
            "Master Data Issue": "Master Data Issue"
        }

        return mapping.get(raw_issue, "User Awareness")

    # ----------------------------------
    # Categorize
    # ----------------------------------
    def categorize(self, text):

        text = self._clean(text)

        scores = {}

        for category, keywords in self.category_keywords.items():

            match_count = sum(
                1 for word in keywords if word in text
            )

            scores[category] = match_count

        # Remove fallback category temporarily
        fallback_score = scores.get("User Awareness", 0)
        scores.pop("User Awareness", None)

        # Pick highest scoring non-fallback category
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        if best_score >= 2:
            confidence = min(best_score / 10, 1.0)
            return best_category, round(confidence, 3)

        # If nothing meaningful matched → fallback
        return "User Awareness", 0.5
