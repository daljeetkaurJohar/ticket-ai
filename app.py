import streamlit as st
import pandas as pd
from predict_engine import classify_ticket  # your existing classifier

st.title("Enterprise Ticket Classification & Dashboard")

uploaded_file = st.file_uploader("Upload Ticket Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    if st.button("Classify Tickets"):
        df["Predicted Category"] = df.apply(lambda row: classify_ticket(row), axis=1)

        st.write(df.head())

        # Save + Download
        df.to_excel("classified_tickets.xlsx", index=False)
        with open("classified_tickets.xlsx", "rb") as f:
            st.download_button("Download Classified Excel", f, file_name="classified_tickets.xlsx")

        # Visualizations
        st.subheader("Category Distribution")
        st.bar_chart(df["Predicted Category"].value_counts())

        df["Group"] = df["Predicted Category"].apply(lambda x: "IT" if "IT" in x else "User")
        st.subheader("IT vs User Split")
        st.bar_chart(df["Group"].value_counts())
