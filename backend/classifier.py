# backend/classifier.py

from categorization_logic import CategorizationLogic

logic = CategorizationLogic("data/issue category.xlsx")

def predict_ticket(text: str):

    category, confidence = logic.categorize(text)

    return {
        "category": category,
        "confidence": confidence,
        "source": "Balanced Text Classifier"
    }
