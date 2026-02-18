
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Enterprise Ticket Intelligence System")

# Upload file
uploaded_file = st.file_uploader(
    "Upload your incident Excel file",
    type=["xlsx"]
)

if uploaded_file is not None:

    # Read uploaded file
    df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully")

    st.subheader("Preview")
    st.dataframe(df)

    # Simple semantic classifier
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")

    categories = [
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

    cat_embeddings = model.encode(categories)

    def classify(row):

        text = " | ".join([str(v) for v in row.values])

        emb = model.encode(text)

        scores = util.cos_sim(emb, cat_embeddings)

        return categories[scores.argmax()]

    if st.button("Classify Tickets"):

        with st.spinner("Classifying..."):

            df["Predicted Category"] = df.apply(classify, axis=1)

        st.success("Classification complete")

        st.subheader("Results")
        st.dataframe(df)

        # Chart
        st.subheader("Category Distribution")

        counts = df["Predicted Category"].value_counts()

        fig, ax = plt.subplots()

        counts.plot(kind="bar", ax=ax)

        st.pyplot(fig)

        # Download button
        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "classified_tickets.csv"
        )
