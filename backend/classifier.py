# backend/classifier.py

import os
import streamlit as st
from categorization_logic import CategorizationLogic


@st.cache_resource
def load_model():

    base_path = os.path.dirname(__file__)
    excel_path = os.path.join(base_path, "..", "data", "issue category.xlsx")

    return CategorizationLogic(excel_path)


logic = load_model()


def predict_ticket(text: str):

    category, confidence = logic.categorize(text)

    return {
        "category": category,
        "confidence": confidence
    }
