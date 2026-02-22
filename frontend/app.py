# frontend/app.py

import streamlit as st
import pandas as pd
import sys
import os
import io
import re
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
    # Predict Categories + Intelligent Executive Summary
    # ---------------------------------------------------
    import re
    
    predictions = []
    confidences = []
    executive_summaries = []
    
    # Columns that are clearly metadata (ignore these)
    ignore_columns = [
        "Resolved on",
        "Raised on",
        "Confidence",
        "Predicted Category",
        "Month",
        "Resolved / Raised Date"
    ]
    
    for _, row in df.iterrows():
    
        # Step 1: Read entire row
        text_parts = []
    
        for col in df.columns:
            if col in ignore_columns:
                continue
    
            value = str(row.get(col, "")).strip()
    
            if value and value.lower() != "nan":
                text_parts.append(value)
    
        full_text = " ".join(text_parts)
    
        # Step 2: Clean timestamps
        full_text = re.sub(r"\d{2}-\d{2}-\d{4}.*?(AM|PM)", "", full_text)
    
        # Step 3: Remove closure noise
        full_text = re.sub(r"resolved.*", "", full_text, flags=re.IGNORECASE)
        full_text = re.sub(r"closing.*", "", full_text, flags=re.IGNORECASE)
        full_text = re.sub(r"completed.*", "", full_text, flags=re.IGNORECASE)
    
        full_text = re.sub(r"\s+", " ", full_text).strip()
    
        if not full_text:
            predictions.append("Insufficient Data")
            confidences.append(0.0)
            executive_summaries.append("Insufficient information available.\nNo issue description found.")
            continue
    
        # Step 4: Predict category
        result = predict_ticket(full_text)
        # If low confidence, mark as Review Required
        category = result["category"]
        confidence = result["confidence"]
        if confidence < 0.25:
            category = "Needs Review"

        predictions.append(result["category"])
        confidences.append(result["confidence"])
    
        # Step 5: Generate professional 2-line executive summary
        line1 = f"{result['category']} impacting business operations."
        line2 = full_text[:250]
    
        executive_summaries.append(f"{line1}\n{line2}")
    
    df["Predicted Category"] = predictions
    df["Confidence"] = confidences
    df["Executive Refined Summary"] = executive_summaries
    # ---------------------------------------------------
    # Professional Executive Summary (2-Line Clean)
    # ---------------------------------------------------
    def refine_summary(row):

        desc = str(row.get("Ticket Description", "")).strip()
        category = str(row.get("Predicted Category", "")).strip()

        # Clean timestamps & noise
        clean_text = re.sub(r"\d{2}-\d{2}-\d{4}.*?(AM|PM)", "", desc)
        clean_text = re.sub(r"-\s*[A-Z\s]+\(Work.*?\)", "", clean_text)
        clean_text = clean_text.strip()

        if not clean_text:
            clean_text = desc[:200]

        line1 = f"{category} impacting business operations."
        line2 = clean_text[:220]

        return f"{line1}\n{line2}"

    df["Executive Refined Summary"] = df.apply(refine_summary, axis=1)


    # ---------------------------------------------------
    # Robust Date & Month Handling
    # ---------------------------------------------------
    
    df.columns = df.columns.str.strip()
    
    possible_date_cols = [
        "Resolved on",
        "Resolved On",
        "resolved on",
        "Raised on",
        "Raised On",
        "raised on"
    ]
    
    date_col = None
    
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
    
        df["Resolved / Raised Date"] = pd.to_datetime(
            df[date_col],
            errors="coerce",
            dayfirst=True
        )
    
        df["Month"] = df["Resolved / Raised Date"].dt.strftime("%B")
    
    else:
        df["Month"] = "Unknown"

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
    # Detailed Ticket View
    # ---------------------------------------------------
    st.markdown("## 📄 Detailed Ticket View")
    st.dataframe(df)


    # ---------------------------------------------------
    # Professional Excel Output
    # ---------------------------------------------------
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        # Detailed Tickets
        df.to_excel(writer, sheet_name="Detailed_Tickets", index=False)

        # Monthly Summary
        monthly_summary = (
            df.groupby(["Month", "Predicted Category"])
            .size()
            .reset_index(name="Ticket Count")
        )
        monthly_summary.to_excel(writer, sheet_name="Monthly_Summary", index=False)

        # Category Overview
        category_summary = (
            df["Predicted Category"]
            .value_counts()
            .reset_index()
        )
        category_summary.columns = ["Category", "Total Tickets"]
        category_summary["Percentage"] = (
            category_summary["Total Tickets"] /
            category_summary["Total Tickets"].sum() * 100
        ).round(2)

        category_summary.to_excel(writer, sheet_name="Category_Overview", index=False)

    output.seek(0)

     st.download_button(
        label="📥 Download Professional Output Report",
        data=output,
        file_name="Ticket_Report_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
