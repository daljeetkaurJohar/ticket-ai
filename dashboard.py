import streamlit as st
import pandas as pd
import io

from predict_engine import classify_file

st.title("Enterprise Ticket Intelligence System")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    st.subheader("Preview")
    st.dataframe(df)

    if st.button("Classify Tickets"):

        with st.spinner("Classifying..."):

            # Save temp file
            input_path = "temp_input.xlsx"
            output_path = "temp_output.xlsx"

            df.to_excel(input_path, index=False)

            classify_file(input_path, output_path)

            result_df = pd.read_excel(output_path)

        st.success("Classification Complete")

        st.subheader("Results")
        st.dataframe(result_df)

        # Summary chart
        st.subheader("Category Distribution")

        counts = result_df["Predicted Category"].value_counts()

        st.bar_chart(counts)

        # Excel download with pivots
        def generate_excel(df):

            output = io.BytesIO()

            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

                df.to_excel(writer, sheet_name="Detailed Data", index=False)

                summary = df["Predicted Category"].value_counts()

                summary.to_excel(writer, sheet_name="Summary")

                if "Assignment Group" in df.columns:

                    pivot = pd.pivot_table(
                        df,
                        index="Assignment Group",
                        columns="Predicted Category",
                        aggfunc="size",
                        fill_value=0
                    )

                    pivot.to_excel(writer, sheet_name="Pivot_Assignment_Group")

            output.seek(0)

            return output

        excel_file = generate_excel(result_df)

        st.download_button(
            "Download Excel Report",
            excel_file,
            file_name="classified_tickets.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
