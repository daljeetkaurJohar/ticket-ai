import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://backend:8000"

st.title("Enterprise Ticket Classification Dashboard")

file = st.file_uploader("Upload Excel", type=["xlsx"])

if file:
    res = requests.post(
        f"{API_URL}/upload",
        files={"file": file}
    )

    batch_id = res.json()["batch_id"]

    st.success(f"Batch Created: {batch_id}")

    if st.button("Start Classification"):
        requests.post(f"{API_URL}/classify/{batch_id}")
        st.success("Classification Started")

if st.button("Load Dashboard"):
    tickets = requests.get(f"{API_URL}/tickets/{batch_id}").json()
    df = pd.DataFrame(tickets)

    st.metric("Total Tickets", len(df))
    st.metric("IT Issues", df["category"].str.startswith("IT").sum())
    st.metric("User Issues", df["category"].str.startswith("User").sum())

    st.subheader("Category Breakdown")
    df["category"].value_counts().plot(kind="bar")
    st.pyplot(plt)
