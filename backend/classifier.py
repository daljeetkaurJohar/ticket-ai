# backend/classifier.py

import streamlit as st
from categorization_logic import CategorizationLogic


@st.cache_resource
def load_model():
    return CategorizationLogic("data/issue category.xlsx")


logic = load_model()


def predict_ticket(text: str):

    category, confidence = logic.categorize(text)

    return {
        "category": category,
        "confidence": confidence,
        "source": "Balanced Text Classifier"
    }
