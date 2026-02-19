import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from classifier import auto_label

st.set_page_config(layout="wide")

st.title("Enterprise Ticket Classification System")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())

    if st.button("Run Classification"):

        df["combined_text"] = (
            df.get("Ticket Summary", "").fillna('') + " " +
            df.get("Ticket Details", "").fillna('') + " " +
            df.get("Work Notes", "").fillna('')
        )

        df["Auto_Category"] = df["combined_text"].apply(auto_label)

        st.success("Classification Completed!")

        # DASHBOARD SECTION
        st.subheader("Category Distribution")

        fig1 = plt.figure()
        df["Auto_Category"].value_counts().plot(kind="bar")
        plt.xticks(rotation=45)
        plt.title("Ticket Distribution by Category")
        st.pyplot(fig1)

        # IT vs User Split
        st.subheader("IT vs User Split")

        df["Group"] = df["Auto_Category"].apply(
            lambda x: "IT" if x.startswith("IT") else "User"
        )

        fig2 = plt.figure()
        df["Group"].value_counts().plot(kind="pie", autopct="%1.1f%%")
        plt.ylabel("")
        plt.title("IT vs User Issues")
        st.pyplot(fig2)

        # Download
        output_file = "classified_output.xlsx"
        df.to_excel(output_file, index=False)

        with open(output_file, "rb") as f:
            st.download_button(
                label="Download Classified File",
                data=f,
                file_name="classified_output.xlsx"
            )
