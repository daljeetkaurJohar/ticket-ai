import pandas as pd
import os
import json

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


# ==========================
# Create folders
# ==========================

os.makedirs("model/fine_tuned", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ==========================
# Load data
# ==========================

df = pd.read_excel("data/incoming_tickets.xlsx")

df.columns = df.columns.str.strip()

df = df.dropna(subset=["ISSUE CAT"])


# ==========================
# Build text context
# ==========================

def build_text(row):

    parts = []

    cols = [
        "Ticket Summary",
        "Ticket Details",
        "Problem",
        "Cause",
        "Assignment Group"
    ]

    for col in cols:

        if col in row and pd.notna(row[col]):

            parts.append(str(row[col]))

    return " | ".join(parts)


df["text"] = df.apply(build_text, axis=1)


# ==========================
# Create label map
# ==========================

categories = df["ISSUE CAT"].unique()

label_map = {cat: i for i, cat in enumerate(categories)}

reverse_map = {i: cat for cat, i in label_map.items()}

# Save label map
with open("model/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)


# ==========================
# Prepare training examples
# ==========================

train_examples = []

for _, row in df.iterrows():

    text = row["text"]

    label = row["ISSUE CAT"]

    train_examples.append(
        InputExample(texts=[text, label])
    )


# ==========================
# Load base model
# ==========================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================
# Train
# ==========================

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=16
)

train_loss = losses.CosineSimilarityLoss(model)


print("Training started...")

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=4,
    warmup_steps=100
)


# ==========================
# Save model
# ==========================

model.save("model/fine_tuned")

print("Model saved successfully!")
print("Check folder: model/fine_tuned")
