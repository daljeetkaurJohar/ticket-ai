# backend/categorization_logic.py

import re


class CategorizationLogic:

    def __init__(self):
        self.rules = self._build_rules()

    def _clean(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    def _build_rules(self):

        return {

            "IT : System Issue": [
                "integration", "not flowing", "not reflecting",
                "system error", "technical failure"
            ],

            "IT : Access": [
                "access", "login", "permission",
                "authorization", "unable to login"
            ],

            "IT : Master Data": [
                "backend master", "system master data",
                "data load failed"
            ],

            "User : Mappings Missing": [
                "mapping", "not mapped",
                "missing mapping"
            ],

            "User : Master Data": [
                "rate missing", "material not visible",
                "bulk rate", "new material"
            ],

            "User : Business Logic Issue": [
                "excel mismatch", "logic difference",
                "calculation wrong", "version issue"
            ],

            "Master Data Issue": [
                "master data mismatch",
                "master data error"
            ],

            "User Awareness": [
                "how to", "clarification",
                "guidance", "cannot understand"
            ]
        }

    def categorize(self, text):

        text = self._clean(text)

        scores = {}

        for category, keywords in self.rules.items():

            score = 0

            for keyword in keywords:
                if keyword in text:
                    score += 1

            scores[category] = score

        # Remove fallback category temporarily
        fallback_score = scores["User Awareness"]
        del scores["User Awareness"]

        # Find best non-fallback category
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        # If meaningful match found
        if best_score >= 1:
            confidence = min(best_score / 3, 1.0)
            return best_category, round(confidence, 3)

        # If nothing matched strongly → check if awareness matched
        if fallback_score >= 1:
            return "User Awareness", 0.7

        # Final fallback
        return "User Awareness", 0.4
