import streamlit as st
import pandas as pd
import plotly.express as px
import io
import os

from predict_engine import classify_file


# =====================================
# Page config
# =====================================

st.set_page_config(
    page_title="Enterprise Ticket Intelligence",
    layout="wide"
)

st.title("Enterprise Ticket Intelligence Dashboard")


# =====================================
# Excel Report Generator with Charts
# =====================================

def generate_excel_with_charts(df):

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

        workbook = writer.book

        # Sheet 1: Detailed Data
        df.to_excel(writer, sheet_name="Detailed Data", index=False)

        detail_sheet = writer.sheets["Detailed Data"]

        for i, col in enumerate(df.columns):
            detail_sheet.set_column(i, i, 25)


        # Sheet 2: Summary
        summary = df["Predicted Category"].value_counts().reset_index()

        summary.columns = ["Category", "Count"]

        summary.to_excel(writer, sheet_name="Summary", index=False)


        # Sheet 3: Charts
        chart_sheet = workbook.add_worksheet("Charts")

        num_rows = len(summary)


        # Bar Chart
        bar_chart = workbook.add_chart({"type": "column"})

        bar_chart.add_series({
            "name": "Category Distribution",
            "categories": ["Summary", 1, 0, num_rows, 0],
            "values": ["Summary", 1, 1, num_rows, 1],
            "data_labels": {"value": True},
        })

        bar_chart.set_title({"name": "Ticket Category Distribution"})

        bar_chart.set_x_axis({"name": "Category"})
        bar_chart.set_y_axis({"name": "Ticket Count"})

        chart_sheet.insert_chart("B2", bar_chart, {"x_scale": 2, "y_scale": 2})


        # Pie Chart
        pie_chart = workbook.add_chart({"type": "pie"})

        pie_chart.add_series({
            "name": "Category Share",
            "categories": ["Summary", 1, 0, num_rows, 0],
            "values": ["Summary", 1, 1, num_rows, 1],
            "data_labels": {"percentage": True},
        })

        pie_chart.set_title({"name": "Category Share"})

        chart_sheet.insert_chart("B25", pie_chart, {"x_scale": 2, "y_scale": 2})


        # Assignment Group Chart
        if "Assignment Group" in df.columns:

            ag = df["Assignment Group"].value_counts().reset_index()

            ag.columns = ["Assignment Group", "Count"]

            ag.to_excel(writer, sheet_name="Assignment Summary", index=False)

            ag_chart = workbook.add_chart({"type": "column"})

            ag_chart.add_series({
                "name": "Tickets by Assignment Group",
                "categories": ["Assignment Summary", 1, 0, len(ag), 0],
                "values": ["Assignment Summary", 1, 1, len(ag), 1],
            })

            ag_chart.set_title({"name": "Tickets by Assignment Group"})

            chart_sheet.insert_chart(
                "B48",
                ag_chart,
                {"x_scale": 2, "y_scale": 2}
            )

    output.seek(0)

    return output


# =====================================
# File Upload
# =====================================

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    st.subheader("Preview")

    st.dataframe(df, width="stretch")


    # =====================================
    # Classification Button
    # =====================================

    if st.button("Run Classification"):

        input_path = "temp_input.xlsx"
        output_path = "temp_output.xlsx"

        df.to_excel(input_path, index=False)

        if not os.path.exists("data/category_examples.json"):

            st.error("category_examples.json missing in data folder")
            st.stop()

        with st.spinner("Classifying tickets..."):

            classify_file(input_path, output_path)

        result_df = pd.read_excel(output_path)

        st.success("Classification Complete")


        # =====================================
        # KPI Metrics
        # =====================================

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


        # =====================================
        # Charts in App
        # =====================================

        counts = result_df["Predicted Category"].value_counts()

        st.subheader("Category Distribution")

        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index
        )

        st.plotly_chart(fig, width="stretch")


        st.subheader("Category Share")

        pie = px.pie(
            names=counts.index,
            values=counts.values
        )

        st.plotly_chart(pie, width="stretch")


        # =====================================
        # Excel Download with Charts
        # =====================================

        excel_file = generate_excel_with_charts(result_df)

        st.download_button(
            "Download Excel Dashboard with Charts",
            excel_file,
            file_name="ticket_dashboard.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


        # =====================================
        # Show Results Table
        # =====================================

        st.subheader("Classified Results")

        st.dataframe(result_df, width="stretch")
