from predict_engine import classify_file
import os
import pandas as pd

INPUT = "data/incoming_tickets.xlsx"
OUTPUT = "output/classified_tickets.xlsx"

def generate_full_report(df, output_file):

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:

        workbook = writer.book

        # Sheet 1: Detailed Data
        df.to_excel(writer, sheet_name="Detailed Data", index=False)

        worksheet = writer.sheets["Detailed Data"]

        # Auto-adjust column width
        for i, col in enumerate(df.columns):
            worksheet.set_column(i, i, 25)


        # Sheet 2: Summary
        summary = df["Predicted Category"].value_counts().reset_index()
        summary.columns = ["Category", "Count"]

        summary.to_excel(writer, sheet_name="Summary", index=False)


        # Sheet 3: Pivot Planning Area
        if "Planning Area" in df.columns:

            pivot1 = pd.pivot_table(
                df,
                index="Planning Area",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )

            pivot1.to_excel(writer, sheet_name="Pivot_Planning_Area")


        # Sheet 4: Pivot Location
        if "Location" in df.columns:

            pivot2 = pd.pivot_table(
                df,
                index="Location",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )

            pivot2.to_excel(writer, sheet_name="Pivot_Location")


        # Sheet 5: Pivot IT Lead
        if "IT Lead" in df.columns:

            pivot3 = pd.pivot_table(
                df,
                index="IT Lead",
                columns="Predicted Category",
                aggfunc="size",
                fill_value=0
            )

            pivot3.to_excel(writer, sheet_name="Pivot_IT_Lead")


if __name__ == "__main__":

    if not os.path.exists(INPUT):

        print("Input file not found:", INPUT)

    else:

        print("Running classification...")

        # Step 1: classify tickets
        classify_file(INPUT, OUTPUT)

        # Step 2: load classified file
        df = pd.read_excel(OUTPUT)

        # Step 3: generate full Excel report with pivots
        generate_full_report(df, OUTPUT)

        print("Full report generated:", OUTPUT)
