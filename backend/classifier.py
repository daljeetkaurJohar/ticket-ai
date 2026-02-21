import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load Issue Category Excel
# -----------------------------
def load_category_logic():
    xls = pd.ExcelFile("data/issue category.xlsx")
    df_list = []

    for sheet in xls.sheet_names:
        temp = pd.read_excel(xls, sheet)
        df_list.append(temp)

    category_df = pd.concat(df_list, ignore_index=True)
    return category_df

category_df = load_category_logic()

# -----------------------------
# Clean Text
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# -----------------------------
# Categorize Ticket
# -----------------------------
def categorize_ticket(short_desc, desc):

    ticket_text = clean_text(short_desc + " " + desc)
    ticket_embedding = model.encode(ticket_text, convert_to_tensor=True)

    best_score = -1
    best_category = "Uncategorized"

    for _, row in category_df.iterrows():

        category_name = row["Issue"]

        combined_text = (
            str(row.get("Definition", "")) + " " +
            str(row.get("Symptoms", "")) + " " +
            str(row.get("Causes", ""))
        )

        combined_clean = clean_text(combined_text)
        category_embedding = model.encode(combined_clean, convert_to_tensor=True)

        score = util.pytorch_cos_sim(ticket_embedding, category_embedding).item()

        if score > best_score:
            best_score = score
            best_category = category_name

    return best_category
