import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r'[^a-zA-Z0-9 ]', '', str(text).lower())

# Load training examples from issue category.xlsx
def load_training_data():
    xls = pd.ExcelFile("data/issue_category.xlsx")
    df_list = []

    for sheet in xls.sheet_names:
        temp = pd.read_excel(xls, sheet)
        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)

    df["combined_text"] = (
        df["Short Description"].astype(str) + " " +
        df["Description"].astype(str)
    ).apply(clean_text)

    return df

training_df = load_training_data()

# Precompute embeddings ONCE
training_embeddings = model.encode(
    training_df["combined_text"].tolist(),
    convert_to_tensor=True
)

def categorize_ticket(short_desc, desc):

    ticket_text = clean_text(short_desc + " " + desc)
    ticket_embedding = model.encode(ticket_text, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(ticket_embedding, training_embeddings)[0]

    best_match_idx = torch.argmax(cosine_scores).item()

    predicted_category = training_df.iloc[best_match_idx]["Issue"]

    confidence = float(cosine_scores[best_match_idx])

    return predicted_category, round(confidence, 3)
