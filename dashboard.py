import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

st.title("Enterprise Ticket Intelligence System")

# Upload file
uploaded_file = st.file_uploader(
    "Upload your incident Excel file",
    type=["xlsx"]
)

# Excel report generator function
def generate_excel_report(df):

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

        workbook = writer.book

        # Sheet 1: Detailed Data
        df.to_excel(writer, sheet_name="Detailed Data", index=False)

        worksheet = writer.sheets["Detailed Data"]

        # Auto width
        for i, col in enumerate(df.columns):
            worksheet.set_column(i, i, 25)

        # Pivot Planning Area
        if "Planning Area" in df.columns:
            pivot1 = pd.pivot_table(
                df,
                index="Planning Area",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )
            pivot1.to_excel(writer, sheet_name="Pivot_Planning_Area")

        # Pivot Location
        if "Location" in df.columns:
            pivot2 = pd.pivot_table(
                df,
                index="Location",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )
            pivot2.to_excel(writer, sheet_name="Pivot_Location")

        # Pivot IT Lead
        if "IT Lead" in df.columns:
            pivot3 = pd.pivot_table(
                df,
                index="IT Lead",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )
            pivot3.to_excel(writer, sheet_name="Pivot_IT_Lead")

        # Summary
        summary = df["Predicted Category"].value_counts()
        summary.to_excel(writer, sheet_name="Summary")

    output.seek(0)

    return output


if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully")

    st.subheader("Preview")
    st.dataframe(df)

    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")

    categories = [
        "IT - System linkage issue",
        "IT - System Access issue",
        "IT - System Version issue",
        "IT - Data entry handholding",
        "IT - Master Data/ mapping issue",
        "User - Mapping missing",
        "User - Master data delayed input",
        "User - Logic changes during ABP",
        "User - Master data incorporation in system",
        "User - System Knowledge Gap",
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

        # Generate Excel report
        excel_file = generate_excel_report(df)

        # Download button
        st.download_button(
            label="Download Excel Report",
            data=excel_file,
            file_name="classified_tickets.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
