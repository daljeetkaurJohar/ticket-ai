import streamlit as st
import pandas as pd
import plotly.express as px
import io

from predict_engine import classify_file


st.set_page_config(
    page_title="Enterprise Ticket Intelligence",
    layout="wide"
)

st.title("Enterprise Ticket Intelligence Dashboard")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    st.subheader("Preview")

    st.dataframe(df, use_container_width=True)

    if st.button("Run Classification"):

        df.to_excel("temp_input.xlsx", index=False)

        classify_file("temp_input.xlsx", "temp_output.xlsx")

        result_df = pd.read_excel("temp_output.xlsx")

        st.success("Classification Complete")


        # KPI
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Tickets", len(result_df))

        col2.metric(
            "IT Issues",
            result_df["Predicted Category"].str.contains("IT").sum()
        )

        col3.metric(
            "User Issues",
            result_df["Predicted Category"].str.contains("User").sum()
        )


        # Bar Chart
        counts = result_df["Predicted Category"].value_counts()

        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            title="Category Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)


        # Pie Chart
        pie = px.pie(
            names=counts.index,
            values=counts.values,
            title="Category Share"
        )

        st.plotly_chart(pie, use_container_width=True)


        # Download Excel
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

            result_df.to_excel(
                writer,
                sheet_name="Detailed Data",
                index=False
            )

        output.seek(0)

        st.download_button(
            "Download Excel Report",
            output,
            file_name="classified_tickets.xlsx"
        )

        st.subheader("Results")

        st.dataframe(result_df, use_container_width=True)
