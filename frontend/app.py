# frontend/app.py

import streamlit as st
import pandas as pd
import sys
import os

# Add backend folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from classifier import predict_ticket

# ----------------------------------
# Streamlit Page Config
# ----------------------------------
st.set_page_config(
    page_title="Enterprise Ticket Classification System",
    layout="wide"
)

st.title("📊 Enterprise Ticket Classification System")
st.markdown("Upload your ticket Excel file for automatic categorization.")

# ----------------------------------
# File Upload
# ----------------------------------
uploaded_file = st.file_uploader("Upload Incident Excel File", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    if df.empty:
        st.warning("Uploaded file is empty.")
        st.stop()

    predictions = []
    confidences = []
    sources = []

    progress_bar = st.progress(0)
    total_rows = len(df)

    for i, (_, row) in enumerate(df.iterrows()):

        # Combine relevant fields safely
        text = " ".join([
            str(row.get("Issue", "")),
            str(row.get("Description", "")),
            str(row.get("Remarks", "")),
            str(row.get("Ticket Description", "")),
            str(row.get("Ticket Summary", "")),
            str(row.get("Ticket Details", ""))
        ])

        result = predict_ticket(text)

        predictions.append(result["category"])
        confidences.append(result["confidence"])
        sources.append(result["source"])

        progress_bar.progress((i + 1) / total_rows)

    # ----------------------------------
    # Add Predictions to DataFrame
    # ----------------------------------
    df["Predicted Category"] = predictions
    df["Confidence"] = confidences
    df["Classification Source"] = sources

    st.success("✅ Classification Complete")

    # ----------------------------------
    # Dashboard Summary
    # ----------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Tickets", len(df))
    col2.metric("Unique Categories", df["Predicted Category"].nunique())
    col3.metric("Avg Confidence", round(df["Confidence"].mean(), 3))

    st.divider()

    # ----------------------------------
    # Show Data
    # ----------------------------------
    st.subheader("📋 Classified Tickets Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # ----------------------------------
    # Category Distribution Chart
    # ----------------------------------
    st.subheader("📊 Category Distribution")

    category_counts = df["Predicted Category"].value_counts()
    st.bar_chart(category_counts)

    # ----------------------------------
    # Download Option
    # ----------------------------------
    output_file = "classified_output.xlsx"
    df.to_excel(output_file, index=False)

    with open(output_file, "rb") as f:
        st.download_button(
            label="📥 Download Classified File",
            data=f,
            file_name=output_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
