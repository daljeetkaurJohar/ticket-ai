from predict_engine import classify_file
import os
import pandas as pd

INPUT = "data/incoming_tickets.xlsx"
OUTPUT = "output/classified_tickets.xlsx"


def generate_full_report(df, output_file):

    # Fix hidden spaces in column names
    df.columns = df.columns.str.strip()

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:

        workbook = writer.book

        # =============================
        # Sheet 1: Detailed Data
        # =============================

        df.to_excel(writer, sheet_name="Detailed Data", index=False)

        detail_sheet = writer.sheets["Detailed Data"]

        for i, col in enumerate(df.columns):
            detail_sheet.set_column(i, i, 25)


        # =============================
        # Sheet 2: Summary
        # =============================

        summary = df["Predicted Category"].value_counts().reset_index()

        summary.columns = ["Category", "Count"]

        summary.to_excel(writer, sheet_name="Summary", index=False)

        summary_sheet = writer.sheets["Summary"]

        summary_sheet.set_column(0, 1, 35)


        # =============================
        # Sheet 3: Pivot Assignment Group
        # =============================

        if "Assignment Group" in df.columns:

            pivot1 = pd.pivot_table(
                df,
                index="Assignment Group",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )

            pivot1["Total"] = pivot1.sum(axis=1)

            pivot1 = pivot1.reset_index()

            pivot1.to_excel(
                writer,
                sheet_name="Pivot_Assignment_Group",
                index=False
            )

            pivot_sheet = writer.sheets["Pivot_Assignment_Group"]

            for i, col in enumerate(pivot1.columns):
                pivot_sheet.set_column(i, i, 30)


        # =============================
        # Sheet 4: Pivot Assigned To
        # =============================

        if "Assigned To" in df.columns:

            pivot2 = pd.pivot_table(
                df,
                index="Assigned To",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )

            pivot2["Total"] = pivot2.sum(axis=1)

            pivot2 = pivot2.reset_index()

            pivot2.to_excel(
                writer,
                sheet_name="Pivot_Assigned_To",
                index=False
            )


        # =============================
        # Sheet 5: Pivot Team
        # =============================

        if "Team" in df.columns:

            pivot3 = pd.pivot_table(
                df,
                index="Team",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )

            pivot3["Total"] = pivot3.sum(axis=1)

            pivot3 = pivot3.reset_index()

            pivot3.to_excel(
                writer,
                sheet_name="Pivot_Team",
                index=False
            )


        # =============================
        # Sheet 6: Pivot Trend
        # =============================

        if "Reported On" in df.columns:

            df["Date"] = pd.to_datetime(
                df["Reported On"],
                errors="coerce"
            ).dt.date

            pivot_trend = pd.pivot_table(
                df,
                index="Date",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )

            pivot_trend = pivot_trend.reset_index()

            pivot_trend.to_excel(
                writer,
                sheet_name="Pivot_Trend",
                index=False
            )


        # =============================
        # Sheet 7: Charts Dashboard
        # =============================

        chart_sheet = workbook.add_worksheet("Charts")

        num_rows = len(summary)


        # Column Chart
        column_chart = workbook.add_chart({"type": "column"})

        column_chart.add_series({
            "name": "Ticket Category Distribution",
            "categories": ["Summary", 1, 0, num_rows, 0],
            "values": ["Summary", 1, 1, num_rows, 1],
            "data_labels": {"value": True},
        })

        column_chart.set_title({
            "name": "Ticket Category Distribution"
        })

        column_chart.set_x_axis({"name": "Category"})
        column_chart.set_y_axis({"name": "Ticket Count"})

        chart_sheet.insert_chart(
            "B2",
            column_chart,
            {"x_scale": 2, "y_scale": 2}
        )


        # Pie Chart
        pie_chart = workbook.add_chart({"type": "pie"})

        pie_chart.add_series({
            "name": "Category Share",
            "categories": ["Summary", 1, 0, num_rows, 0],
            "values": ["Summary", 1, 1, num_rows, 1],
            "data_labels": {"percentage": True},
        })

        pie_chart.set_title({"name": "Category Share"})

        chart_sheet.insert_chart(
            "B25",
            pie_chart,
            {"x_scale": 2, "y_scale": 2}
        )


# =============================
# Main Execution
# =============================

if __name__ == "__main__":

    print("Starting Enterprise Ticket Classification...")

    if not os.path.exists(INPUT):

        print("ERROR: Input file not found:", INPUT)

    else:

        classify_file(INPUT, OUTPUT)

        print("Classification completed.")

        df = pd.read_excel(OUTPUT)

        df.columns = df.columns.str.strip()

        generate_full_report(df, OUTPUT)

        print("Enterprise report generated successfully.")

        print("Output file location:", OUTPUT)
