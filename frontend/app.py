# frontend/app.py

import streamlit as st
import pandas as pd
import sys
import os
import io
from datetime import datetime

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from classifier import predict_ticket


# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Ticket Intelligence Dashboard",
    layout="wide"
)

st.title("📊 Ticket Intelligence & Categorization System")


# ---------------------------------------------------
# File Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader("Upload Ticket Excel File", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    # ---------------------------------------------------
    # Predict Categories
    # ---------------------------------------------------
    predictions = []
    confidences = []

    for _, row in df.iterrows():

            text = " ".join([
                str(row.get("Ticket Description", "")),
                str(row.get("Summary", "")),
                str(row.get("Work notes", "")),
                str(row.get("Remarks", "")),
                str(row.get("Ticket Summary", "")),
                str(row.get("Ticket Details", ""))
            ])

            result = predict_ticket(text)

        predictions.append(result["category"])
        confidences.append(result["confidence"])

    df["Predicted Category"] = predictions
    df["Confidence"] = confidences


    # ---------------------------------------------------
    # Refined Executive Summary
    # ---------------------------------------------------
    def refine_summary(row):

        desc = str(row.get("Ticket Description", ""))
        summary = str(row.get("Summary", ""))
        notes = str(row.get("Work notes", ""))

        refined = f"""
Problem Statement:
{desc.strip()}

Business Context:
{summary.strip()}

Technical / Operational Notes:
{notes.strip()}

Final Classification:
{row.get("Predicted Category","")}

Confidence Score:
{row.get("Confidence","")}
        """.strip()

        return refined

    df["Executive Refined Summary"] = df.apply(refine_summary, axis=1)


    # ---------------------------------------------------
    # Date & Month Handling
    # ---------------------------------------------------
    if "Resolved on" in df.columns:
        df["Resolved / Raised Date"] = pd.to_datetime(df["Resolved on"], errors="coerce")
    elif "Raised on" in df.columns:
        df["Resolved / Raised Date"] = pd.to_datetime(df["Raised on"], errors="coerce")
    else:
        df["Resolved / Raised Date"] = pd.NaT

    df["Month"] = df["Resolved / Raised Date"].dt.strftime("%B")


    # ---------------------------------------------------
    # KPI SECTION
    # ---------------------------------------------------
    st.markdown("## 📌 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    total_tickets = len(df)
    unique_categories = df["Predicted Category"].nunique()
    avg_confidence = round(df["Confidence"].mean(), 3)
    top_category = df["Predicted Category"].value_counts().idxmax()

    col1.metric("Total Tickets", total_tickets)
    col2.metric("Unique Categories", unique_categories)
    col3.metric("Avg Confidence", avg_confidence)
    col4.metric("Top Category", top_category)


    # ---------------------------------------------------
    # DASHBOARD SECTION
    # ---------------------------------------------------
    st.markdown("## 📊 Dashboard Overview")

    # Category Distribution
    st.subheader("Issue Category Distribution")
    category_counts = df["Predicted Category"].value_counts()
    st.bar_chart(category_counts)

    # Month-wise Ticket Count
    st.subheader("Month-wise Ticket Count")
    month_counts = df.groupby("Month").size()
    st.bar_chart(month_counts)

    # Month-wise Percentage Distribution
    st.subheader("Month-wise Issue Percentage")

    month_category = (
        df.groupby(["Month", "Predicted Category"])
          .size()
          .unstack(fill_value=0)
    )

    month_percentage = (
        month_category.div(month_category.sum(axis=1), axis=0) * 100
    )

    st.dataframe(month_percentage.round(2))
    st.bar_chart(month_percentage)


    # ---------------------------------------------------
    # DETAILED DATA VIEW
    # ---------------------------------------------------
    st.markdown("## 📄 Detailed Ticket View")
    st.dataframe(df)


    # ---------------------------------------------------
    # PROFESSIONAL EXCEL OUTPUT
    # ---------------------------------------------------
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        # Detailed Sheet
        df.to_excel(writer, sheet_name="Detailed_Tickets", index=False)

        # Pivot Summary Sheet
        summary_sheet = df.groupby(
            ["Month", "Predicted Category"]
        ).size().reset_index(name="Ticket Count")

        summary_sheet.to_excel(writer, sheet_name="Monthly_Summary", index=False)

        # Category Summary Sheet
        category_summary = df["Predicted Category"].value_counts().reset_index()
        category_summary.columns = ["Category", "Total Tickets"]

        category_summary.to_excel(writer, sheet_name="Category_Summary", index=False)

    output.seek(0)

    st.download_button(
        label="📥 Download Professional Output Report",
        data=output,
        file_name=f"Ticket_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
