# frontend/app.py

import streamlit as st
import pandas as pd
import sys
import os
import io
from datetime import datetime

# Add backend path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from classifier import predict_ticket

st.set_page_config(page_title="Ticket Intelligence Dashboard", layout="wide")

st.title("📊 Ticket Intelligence & Categorization System")

uploaded_file = st.file_uploader("Upload Ticket Excel File", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()

    predictions = []
    confidences = []

    # -----------------------------
    # CLEAN INPUT TEXT (NO WORK NOTES)
    # -----------------------------
    for _, row in df.iterrows():

        text = " ".join([
            str(row.get("Ticket Summary", "")),
            str(row.get("Ticket Description", ""))
        ])

        result = predict_ticket(text)

        predictions.append(result["category"])
        confidences.append(result["confidence"])

    df["Predicted Category"] = predictions
    df["Confidence"] = confidences

    # -----------------------------
    # PROFESSIONAL EXECUTIVE SUMMARY
    # -----------------------------
    def refine_summary(row):

        summary = str(row.get("Ticket Summary", "")).strip()
        description = str(row.get("Ticket Description", "")).strip()
        category = str(row.get("Predicted Category", "")).strip()

        combined = f"{summary} {description}"
        combined = " ".join(combined.split())[:220]

        if not combined:
            combined = "Ticket details insufficient for executive analysis."

        return f"{category} identified impacting operational workflow. {combined}"

    df["Executive Summary"] = df.apply(refine_summary, axis=1)

    # -----------------------------
    # DATE & MONTH HANDLING
    # -----------------------------
    date_col = None
    for possible in ["Closed On", "Reported On", "Resolved on", "Raised on"]:
        if possible in df.columns:
            date_col = possible
            break

    if date_col:
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
        df["Month"] = df["Date"].dt.strftime("%B")
    else:
        df["Month"] = "Unknown"

    # -----------------------------
    # KPI SECTION
    # -----------------------------
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

    # -----------------------------
    # DASHBOARD
    # -----------------------------
    st.markdown("## 📊 Dashboard Overview")

    st.subheader("Issue Category Distribution")
    st.bar_chart(df["Predicted Category"].value_counts())

    st.subheader("Month-wise Ticket Count")
    st.bar_chart(df.groupby("Month").size())

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

    # -----------------------------
    # DETAILED VIEW
    # -----------------------------
    st.markdown("## 📄 Detailed Ticket View")
    st.dataframe(df)

    # -----------------------------
    # PROFESSIONAL REPORT EXPORT
    # -----------------------------
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        df.to_excel(writer, sheet_name="Detailed_Tickets", index=False)

        category_summary = df["Predicted Category"].value_counts().reset_index()
        category_summary.columns = ["Category", "Total Tickets"]

        category_summary["Percentage"] = (
            category_summary["Total Tickets"] /
            category_summary["Total Tickets"].sum() * 100
        ).round(2)

        category_summary.to_excel(writer, sheet_name="Category_Summary", index=False)

        monthly_summary = (
            df.groupby(["Month", "Predicted Category"])
            .size()
            .reset_index(name="Ticket Count")
        )

        monthly_summary.to_excel(writer, sheet_name="Monthly_Summary", index=False)

    output.seek(0)

    st.download_button(
        label="📥 Download Professional Report",
        data=output,
        file_name=f"Ticket_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
