# backend/categorization_logic.py

import re


class CategorizationLogic:

    def __init__(self):
        self.rules = self._build_rules()

    # ----------------------------------
    # Clean text
    # ----------------------------------
    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    # ----------------------------------
    # Explicit deterministic rules
    # ----------------------------------
    def _build_rules(self):

        return {

            "IT : System Issue": {
                "must_have": ["integration", "not flowing", "not reflecting", "error", "failure"],
                "exclude": ["how to", "clarification"]
            },

            "IT : Access": {
                "must_have": ["access", "login", "permission", "authorization"],
                "exclude": []
            },

            "IT : Master Data": {
                "must_have": ["system master data", "backend master", "data load failed"],
                "exclude": []
            },

            "User : Mappings Missing": {
                "must_have": ["mapping", "not mapped", "missing mapping"],
                "exclude": []
            },

            "User : Master Data": {
                "must_have": ["rate missing", "material not visible", "bulk rate", "new material"],
                "exclude": []
            },

            "User : Business Logic Issue": {
                "must_have": ["excel mismatch", "logic difference", "calculation wrong", "version issue"],
                "exclude": []
            },

            "Master Data Issue": {
                "must_have": ["master data mismatch", "master data error"],
                "exclude": []
            },

            "User Awareness": {
                "must_have": ["how to", "clarification", "cannot understand", "guidance"],
                "exclude": []
            }
        }

    # ----------------------------------
    # Categorize
    # ----------------------------------
    def categorize(self, text):

        text = self._clean(text)

        for category, conditions in self.rules.items():

            must_have = conditions["must_have"]
            exclude = conditions["exclude"]

            # Check must-have keywords
            if any(keyword in text for keyword in must_have):

                # Check exclusion
                if not any(ex in text for ex in exclude):
                    return category, 0.9

        # Final fallback
        return "User Awareness", 0.5
