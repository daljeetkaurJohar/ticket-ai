# backend/categorization_logic.py

import pandas as pd
import re


class CategorizationLogic:

    def __init__(self, excel_file):
        self.rules = self._build_rules(excel_file)
        self.category_mapping = self._build_category_mapping()

    # -----------------------------------
    # Clean Text
    # -----------------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # -----------------------------------
    # Load rules row-by-row
    # -----------------------------------
    def _build_rules(self, excel_file):

        df = pd.read_excel(excel_file, sheet_name=0)

        rules = []

        for _, row in df.iterrows():

            issue = str(row.get("Issue", "")).strip()

            rule_text = " ".join([
                str(row.get("Description", "")),
                str(row.get("Issue Description", "")),
                str(row.get("Unnamed: 5", ""))
            ])

            rule_text = self._clean(rule_text)

            if issue and rule_text.strip():
                rules.append({
                    "issue": issue,
                    "rule_text": rule_text
                })

        return rules

    # -----------------------------------
    # Mapping to FINAL 8 categories
    # -----------------------------------
    def _build_category_mapping(self):

        return {
            # IT related
            "System linkage issue": "IT : System Issue",
            "System Access issue": "IT : Access",
            "IT Masterdata issue": "IT : Master Data",

            # User related
            "User knowledge gap": "User Awareness",
            "Mapping missing from user": "User : Mappings Missing",
            "Masterdata - delayed input from user": "User : Master Data",
            "Logic mistakes in excel vs system": "User : Business Logic Issue",
            "Multiple versions issue in excel": "User : Business Logic Issue",

            # Generic master data
            "Master Data Issue": "Master Data Issue",
            "User KT issue": "User Awareness"
        }

    # -----------------------------------
    # Categorize
    # -----------------------------------
    def categorize(self, text):

        text = self._clean(text)

        for rule in self.rules:

            if rule["rule_text"] in text:
                raw_issue = rule["issue"]

                final_category = self.category_mapping.get(
                    raw_issue,
                    "User Awareness"
                )

                return final_category, 0.95

        return "User Awareness", 0.5
