import pandas as pd
import json
import os

INPUT_FILE = "data/incoming_tickets.xlsx"

OUTPUT_FILE = "data/category_examples.json"

# How many examples per category
EXAMPLES_PER_CATEGORY = 20


def build_context(row):
    """
    Combine all useful columns into one semantic text
    """
    parts = []

    important_cols = [
        "Ticket Summary",
        "Ticket Details",
        "Problem",
        "Cause",
        "Work notes",
        "Assignment Group",
        "Team"
    ]

    for col in important_cols:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]))

    return " | ".join(parts)


def generate_examples():

    df = pd.read_excel(INPUT_FILE)

    df.columns = df.columns.str.strip()

    # Use your true category column if available
    if "ISSUE CAT" in df.columns:
        category_column = "ISSUE CAT"
    elif "Predicted Category" in df.columns:
        category_column = "Predicted Category"
    else:
        raise Exception("No category column found")

    examples = {}

    for category, group in df.groupby(category_column):

        texts = []

        for _, row in group.iterrows():

            context = build_context(row)

            if context.strip():
                texts.append(context)

        # Take top N examples
        examples[category] = texts[:EXAMPLES_PER_CATEGORY]

    # Save JSON
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print("Examples file created:", OUTPUT_FILE)


if __name__ == "__main__":

    generate_examples()
