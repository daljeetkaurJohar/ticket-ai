# backend/categorization_logic.py

import pandas as pd
import re


class CategorizationLogic:

    def __init__(self, excel_file):
        self.rules = self._extract_rules(excel_file)

    # -----------------------------
    # Clean Text
    # -----------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # -----------------------------
    # Extract structured rules
    # -----------------------------
    def _extract_rules(self, excel_file):

        df = pd.read_excel(excel_file, sheet_name=0)

        rule_dict = {}

        for _, row in df.iterrows():

            issue = str(row.get("Issue", "")).strip()

            rule_text = " ".join([
                str(row.get("Description", "")),
                str(row.get("Issue Description", "")),
                str(row.get("Unnamed: 5", ""))
            ])

            rule_text = self._clean(rule_text)

            keywords = [
                word.strip()
                for word in rule_text.split()
                if len(word) > 3
            ]

            if issue not in rule_dict:
                rule_dict[issue] = set()

            rule_dict[issue].update(keywords)

        return rule_dict

    # -----------------------------
    # Deterministic Categorization
    # -----------------------------
    def categorize(self, text):

        text = self._clean(text)

        # PRIORITY ORDER
        priority_order = [
            "System linkage issue",
            "Mapping missing from user",
            "Multiple versions issue in excel",
            "Masterdata - delayed input from user",
            "Logic mistakes in excel vs system",
            "System Access issue",
            "User KT issue",
            "User knowledge gap"
        ]

        for category in priority_order:

            if category not in self.rules:
                continue

            keywords = self.rules[category]

            # Must match at least 2 keywords
            matches = sum(1 for word in keywords if word in text)

            if matches >= 2:
                confidence = min(matches / 5, 1.0)
                return category, round(confidence, 3)

        return "Needs Manual Review", 0.0
