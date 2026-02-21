# backend/categorization_logic.py

import pandas as pd
import re
from collections import defaultdict


class CategorizationLogic:

    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.issue_reference_map = self._build_reference_engine()

    # ------------------------------
    # Clean Text
    # ------------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # ------------------------------
    # Build Reference Engine
    # ------------------------------
    def _build_reference_engine(self):

        xls = pd.ExcelFile(self.excel_file)
        sheets = xls.sheet_names

        issue_map = defaultdict(list)

        for sheet in sheets:
            df = pd.read_excel(xls, sheet)

            if "Issue" not in df.columns:
                continue

            for _, row in df.iterrows():

                issue = str(row.get("Issue", "")).strip()

                combined_text = " ".join([
                    str(row.get("Description", "")),
                    str(row.get("Issue Description", "")),
                    str(row.get("Remarks", "")),
                    str(row.get("Unnamed: 5", ""))
                ])

                combined_text = self._clean(combined_text)

                if issue and combined_text.strip():
                    issue_map[issue].append(combined_text)

        return issue_map

    # ------------------------------
    # Categorize
    # ------------------------------
    def categorize(self, text):

        text = self._clean(text)

        best_issue = None
        best_score = 0

        for issue, reference_texts in self.issue_reference_map.items():

            score = 0

            for ref in reference_texts:
                common_words = set(text.split()) & set(ref.split())
                score += len(common_words)

            if score > best_score:
                best_score = score
                best_issue = issue

        if best_issue:
            confidence = min(best_score / 20, 1.0)
            return best_issue, round(confidence, 3)

        return "Needs Manual Review", 0.0
