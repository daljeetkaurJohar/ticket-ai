
from predict_engine import classify_file
import os

INPUT = "data/incoming_tickets.xlsx"
OUTPUT = "output/classified_tickets.xlsx"

if __name__ == "__main__":
    if not os.path.exists(INPUT):
        print("Input file not found:", INPUT)
    else:
        classify_file(INPUT, OUTPUT)

import pandas as pd

# After classification
writer = pd.ExcelWriter("output/classified_tickets.xlsx", engine="xlsxwriter")

# Sheet 1: Detailed Data
df.to_excel(writer, sheet_name="Detailed Data", index=False)

# Sheet 2: Summary
summary = df["Predicted Category"].value_counts()
summary.to_excel(writer, sheet_name="Summary")

# Sheet 3: Pivot Planning Area
pivot1 = pd.pivot_table(
    df,
    index="Planning Area",
    columns="Predicted Category",
    aggfunc="size",
    fill_value=0
)

pivot1.to_excel(writer, sheet_name="Pivot Planning Area")

# Sheet 4: Pivot Location
pivot2 = pd.pivot_table(
    df,
    index="Location",
    columns="Predicted Category",
    aggfunc="size",
    fill_value=0
)

pivot2.to_excel(writer, sheet_name="Pivot Location")

writer.close()
