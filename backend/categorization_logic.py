# backend/categorization_logic.py

import pandas as pd
import re

class CategorizationLogic:

    def __init__(self, excel_file):
        self.df = pd.read_excel(excel_file)
        self.rules = self._build_rules()

    def _clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    def _build_rules(self):
        rules = {}

        for _, row in self.df.iterrows():

            category = str(row["Issue"]).strip()

            combined_text = " ".join([
                str(row.get("Description", "")),
                str(row.get("Issue Description", "")),
                str(row.get("Unnamed: 5", ""))
            ])

            combined_text = self._clean_text(combined_text)
            keywords = set(combined_text.split())

            if category not in rules:
                rules[category] = set()

            rules[category].update(keywords)

        return rules

    def categorize(self, text):

        text = self._clean_text(text)
        words = set(text.split())

        best_category = None
        best_score = 0

        for category, keywords in self.rules.items():

            score = len(words.intersection(keywords))

            if score > best_score:
                best_score = score
                best_category = category

        if best_category:
            confidence = min(best_score / 10, 1.0)
            return best_category, round(confidence, 3)

        return "Needs Manual Review", 0.0
