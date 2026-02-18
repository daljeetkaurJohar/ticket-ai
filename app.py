from predict_engine import classify_file
import os
import pandas as pd

INPUT = "data/incoming_tickets.xlsx"
OUTPUT = "output/classified_tickets.xlsx"


def generate_full_report(df, output_file):

    # CRITICAL FIX: remove hidden spaces from column names
    df.columns = df.columns.str.strip()

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:

        workbook = writer.book

        # =============================
        # Sheet 1: Detailed Data
        # =============================

        df.to_excel(writer, sheet_name="Detailed Data", index=False)

        worksheet = writer.sheets["Detailed Data"]

        for i, col in enumerate(df.columns):
            worksheet.set_column(i, i, 25)


        # =============================
        # Sheet 2: Summary
        # =============================

        summary = df["Predicted Category"].value_counts().reset_index()
        summary.columns = ["Category", "Count"]

        summary.to_excel(writer, sheet_name="Summary", index=False)


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

            pivot1.to_excel(writer, sheet_name="Pivot_Assignment_Group")


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

            pivot2.to_excel(writer, sheet_name="Pivot_Assigned_To")


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

            pivot3.to_excel(writer, sheet_name="Pivot_Team")


        # =============================
        # Sheet 6: Trend Pivot
        # =============================

        if "Reported On" in df.columns:

            df["Date"] = pd.to_datetime(df["Reported On"], errors="coerce").dt.date

            pivot_trend = pd.pivot_table(
                df,
                index="Date",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )

            pivot_trend.to_excel(writer, sheet_name="Pivot_Trend")


        # =============================
        # Sheet 7: Charts
        # =============================

        chart_sheet = workbook.add_worksheet("Charts")

        chart = workbook.add_chart({"type": "column"})

        chart.add_series({
            "name": "Category Distribution",
            "categories": ["Summary", 1, 0, len(summary), 0],
            "values": ["Summary", 1, 1, len(summary), 1],
        })

        chart.set_title({"name": "Ticket Category Distribution"})
        chart_sheet.insert_chart("B2", chart)


        pie_chart = workbook.add_chart({"type": "pie"})

        pie_chart.add_series({
            "categories": ["Summary", 1, 0, len(summary), 0],
            "values": ["Summary", 1, 1, len(summary), 1],
        })

        pie_chart.set_title({"name": "Category Share"})
        chart_sheet.insert_chart("B20", pie_chart)



if __name__ == "__main__":

    if not os.path.exists(INPUT):

        print("Input file not found:", INPUT)

    else:

        print("Running classification...")

        classify_file(INPUT, OUTPUT)

        print("Classification complete.")

        df = pd.read_excel(OUTPUT)

        # CRITICAL FIX HERE ALSO
        df.columns = df.columns.str.strip()

        generate_full_report(df, OUTPUT)

        print("Enterprise report generated successfully.")
        print("Output file:", OUTPUT)
