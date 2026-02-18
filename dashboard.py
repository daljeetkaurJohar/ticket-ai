import streamlit as st
import pandas as pd
import plotly.express as px
import io

from predict_engine import classify_file

st.set_page_config(
    page_title="Enterprise Ticket Intelligence",
    layout="wide"
)

# Custom CSS for colors
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.kpi-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


st.title("Enterprise Ticket Intelligence Dashboard")

uploaded_file = st.file_uploader(
    "Upload Ticket Excel File",
    type=["xlsx"]
)

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    st.subheader("Data Preview")
    st.dataframe(df, use_container_width=True)


    if st.button("Run Classification"):

        with st.spinner("Processing..."):

            df.to_excel("temp_input.xlsx", index=False)

            classify_file("temp_input.xlsx", "temp_output.xlsx")

            result_df = pd.read_excel("temp_output.xlsx")


        st.success("Classification Complete")


        # ======================
        # KPI SECTION
        # ======================

        total = len(result_df)

        it_count = result_df[
            result_df["Predicted Category"].str.contains("IT")
        ].shape[0]

        user_count = result_df[
            result_df["Predicted Category"].str.contains("User")
        ].shape[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Tickets", total)
        col2.metric("IT Issues", it_count)
        col3.metric("User Issues", user_count)


        # ======================
        # Bar Chart
        # ======================

        st.subheader("Category Distribution")

        counts = result_df["Predicted Category"].value_counts()

        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.values,
            color_continuous_scale="Blues",
            title="Tickets by Category"
        )

        st.plotly_chart(fig, use_container_width=True)


        # ======================
        # Pie Chart
        # ======================

        st.subheader("Category Share")

        pie = px.pie(
            names=counts.index,
            values=counts.values,
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        st.plotly_chart(pie, use_container_width=True)


        # ======================
        # Assignment Group Chart
        # ======================

        if "Assignment Group" in result_df.columns:

            st.subheader("Tickets by Assignment Group")

            ag = result_df["Assignment Group"].value_counts()

            fig2 = px.bar(
                x=ag.index,
                y=ag.values,
                color=ag.values,
                color_continuous_scale="Viridis"
            )

            st.plotly_chart(fig2, use_container_width=True)


        # ======================
        # Download Excel Button
        # ======================

        def generate_excel(df):

            output = io.BytesIO()

            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

                df.to_excel(
                    writer,
                    sheet_name="Detailed Data",
                    index=False
                )

                summary = df["Predicted Category"].value_counts()

                summary.to_excel(
                    writer,
                    sheet_name="Summary"
                )

            output.seek(0)

            return output


        excel = generate_excel(result_df)

        st.download_button(
            "Download Excel Report",
            excel,
            "ticket_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


        st.subheader("Classified Data")
        st.dataframe(result_df, use_container_width=True)
