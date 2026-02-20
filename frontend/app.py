import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

st.title("Enterprise Ticket Classifier")

CATEGORIES = [
    "IT - System linkage issue",
    "IT - System Access issue",
    "IT – System Version issue",
    "IT – Data entry handholding",
    "IT – Master Data/ mapping issue",
    "User - Mapping missing",
    "User – Master data delayed input",
    "User - Logic changes during ABP",
    "User – Master data incorporation in system",
    "User – System Knowledge Gap",
    "User - Logic mistakes in excel vs system",
    "User - Multiple versions issue in excel"
]

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
category_embeddings = model.encode(CATEGORIES)

def classify(text):
    emb = model.encode([text])
    sim = cosine_similarity(emb, category_embeddings)
    idx = np.argmax(sim)
    return CATEGORIES[idx], float(sim[0][idx])

file = st.file_uploader("Upload Excel", type=["xlsx"])

if file:
    df = pd.read_excel(file)

    if "Description" not in df.columns:
        st.error("Excel must contain 'Description' column")
    else:
        categories = []
        confidences = []

        with st.spinner("Classifying tickets..."):
            for desc in df["Description"]:
                cat, conf = classify(str(desc))
                categories.append(cat)
                confidences.append(conf)

        df["AI_Category"] = categories
        df["Confidence"] = confidences

        st.success("Classification Complete")

        st.metric("Total Tickets", len(df))
        st.metric("IT Issues", df["AI_Category"].str.startswith("IT").sum())
        st.metric("User Issues", df["AI_Category"].str.startswith("User").sum())

        st.bar_chart(df["AI_Category"].value_counts())

        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)

        st.download_button(
            "Download Classified Excel",
            output,
            file_name="classified.xlsx"
        )
