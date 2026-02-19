import streamlit as st
import pandas as pd
from predict_engine import EliteClassifier

st.title("Elite Enterprise Ticket Intelligence")

clf = EliteClassifier()

uploaded = st.file_uploader("Upload Excel", type=["xlsx"])

if uploaded:
    df = pd.read_excel(uploaded)

    if st.button("Run Elite Classification"):
        preds = []
        confs = []

        for _, row in df.iterrows():
            text = " | ".join([str(v) for v in row if pd.notna(v)])
            p, c = clf.predict(text)
            preds.append(p)
            confs.append(c)

        df["Predicted Category"] = preds
        df["Confidence"] = confs

        st.dataframe(df)

        df.to_excel("elite_output.xlsx", index=False)
        st.download_button(
            "Download Results",
            open("elite_output.xlsx", "rb"),
            file_name="elite_output.xlsx"
        )
