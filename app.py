import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from classifier import auto_label  # your rule engine

st.set_page_config(layout="wide")
st.title("Enterprise Ticket Intelligence System")

# Tabs
tab1, tab2 = st.tabs(["📂 Classify Tickets", "📊 Dashboard"])

# ----------------------------
# TAB 1 – CLASSIFICATION
# ----------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload Ticket Excel File", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        st.subheader("Preview Uploaded Data")
        st.dataframe(df.head())

        if st.button("Run Automatic Classification"):

            df["combined_text"] = (
                df.get("Ticket Summary", "").fillna('') + " " +
                df.get("Ticket Details", "").fillna('') + " " +
                df.get("Work Notes", "").fillna('')
            )

            df["Predicted Category"] = df["combined_text"].apply(auto_label)

            st.success("Classification Completed Successfully!")

            st.dataframe(df.head())

            # Save output
            output_file = "classified_output.xlsx"
            df.to_excel(output_file, index=False)

            with open(output_file, "rb") as f:
                st.download_button(
                    label="⬇ Download Classified File",
                    data=f,
                    file_name="classified_output.xlsx"
                )

            # Store for dashboard
            st.session_state["classified_df"] = df


# ----------------------------
# TAB 2 – DASHBOARD
# ----------------------------
with tab2:
    if "classified_df" in st.session_state:

        df = st.session_state["classified_df"]

        st.subheader("Category Distribution")

        fig1 = plt.figure()
        df["Predicted Category"].value_counts().plot(kind="bar")
        plt.xticks(rotation=45)
        plt.title("Ticket Distribution by Category")
        st.pyplot(fig1)

        # IT vs User Split
        st.subheader("IT vs User Split")

        df["Group"] = df["Predicted Category"].apply(
            lambda x: "IT" if x.startswith("IT") else "User"
        )

        fig2 = plt.figure()
        df["Group"].value_counts().plot(kind="pie", autopct="%1.1f%%")
        plt.ylabel("")
        plt.title("IT vs User Issues")
        st.pyplot(fig2)

        # Confidence placeholder (for future ML integration)
        st.info("Confidence scoring module can be added in next upgrade.")

    else:
        st.warning("Please classify tickets first in the 'Classify Tickets' tab.")
