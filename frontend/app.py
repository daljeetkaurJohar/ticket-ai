import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

st.set_page_config(page_title="Enterprise Ticket Classifier", layout="wide")

st.title("Enterprise Ticket Classifier")

# -------------------------
# Category List
# -------------------------
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

# -------------------------
# Load Model (cached)
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
category_embeddings = model.encode(CATEGORIES)

# -------------------------
# Classification Function
# -------------------------
def classify(text):
    emb = model.encode([text])
    sim = cosine_similarity(emb, category_embeddings)
    idx = np.argmax(sim)
    return CATEGORIES[idx], float(sim[0][idx])

# -------------------------
# File Upload
# -------------------------
file = st.file_uploader("Upload Excel File", type=["xlsx"])

if file:
    df = pd.read_excel(file)

    # -------------------------
    # Auto-detect description columns
    # -------------------------
    possible_cols = [
        "Description",
        "Ticket Details",
        "Ticket Summary",
        "Work notes",
        "Problem",
        "Solution"
    ]

    available_cols = [col for col in df.columns if col.strip() in possible_cols]

    if not available_cols:
        st.error("No suitable description column found.")
        st.stop()

    st.success(f"Using columns: {', '.join(available_cols)}")

    # Combine selected columns into one text field
    df["__combined_text__"] = df[available_cols].fillna("").agg(" ".join, axis=1)

    # -------------------------
    # Classification
    # -------------------------
    categories = []
    confidences = []

    with st.spinner("Classifying tickets..."):
        for text in df["__combined_text__"]:
            cat, conf = classify(str(text))
            categories.append(cat)
            confidences.append(conf)

    df["AI_Category"] = categories
    df["Confidence"] = confidences

    # -------------------------
    # Dashboard Metrics
    # -------------------------
    st.subheader("Dashboard")

    total = len(df)
    it_count = df["AI_Category"].str.startswith("IT").sum()
    user_count = df["AI_Category"].str.startswith("User").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tickets", total)
    col2.metric("IT Issues", it_count)
    col3.metric("User Issues", user_count)

    # Category breakdown chart
    st.subheader("Category Breakdown")
    st.bar_chart(df["AI_Category"].value_counts())

    # -------------------------
    # Low Confidence Highlight
    # -------------------------
    low_conf_df = df[df["Confidence"] < 0.60]

    if not low_conf_df.empty:
        st.warning(f"{len(low_conf_df)} tickets have low confidence (<0.60)")
        st.dataframe(low_conf_df)

    # -------------------------
    # Show Classified Data
    # -------------------------
    st.subheader("Classified Tickets")
    st.dataframe(df.drop(columns=["__combined_text__"]))

    # -------------------------
    # Download Excel
    # -------------------------
    output = BytesIO()
    df.drop(columns=["__combined_text__"]).to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        "Download Classified Excel",
        output,
        file_name="classified_tickets.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
