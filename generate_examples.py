import pandas as pd
import json
import os

INPUT_FILE = "data/incoming_tickets.xlsx"

OUTPUT_FILE = "data/category_examples.json"

EXAMPLES_PER_CATEGORY = 25


def build_context(row):

    parts = []

    for col in row.index:

        if pd.notna(row[col]):

            parts.append(str(row[col]))

    return " | ".join(parts)


def generate_examples():

    df = pd.read_excel(INPUT_FILE)

    df.columns = df.columns.str.strip()

    if "ISSUE CAT" in df.columns:

        category_col = "ISSUE CAT"

    elif "Predicted Category" in df.columns:

        category_col = "Predicted Category"

    else:

        raise Exception(
            "No category column found. Must have ISSUE CAT or Predicted Category"
        )

    examples = {}

    for category, group in df.groupby(category_col):

        texts = []

        for _, row in group.iterrows():

            context = build_context(row)

            texts.append(context)

        examples[category] = texts[:EXAMPLES_PER_CATEGORY]

    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(examples, f, indent=2, ensure_ascii=False)

    print("Created:", OUTPUT_FILE)


if __name__ == "__main__":

    generate_examples()
