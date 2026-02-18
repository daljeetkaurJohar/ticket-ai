import pandas as pd
import torch

from datasets import Dataset
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

import numpy as np
import os

# ==========================
# Load your data
# ==========================

df = pd.read_excel("data/incoming_tickets.xlsx")

df.columns = df.columns.str.strip()

df = df.dropna(subset=["ISSUE CAT"])


# ==========================
# Build text context
# ==========================

def build_text(row):

    parts = []

    important_cols = [
        "Ticket Summary",
        "Ticket Details",
        "Problem",
        "Cause",
        "Assignment Group"
    ]

    for col in important_cols:

        if col in row and pd.notna(row[col]):

            parts.append(str(row[col]))

    return " | ".join(parts)


df["text"] = df.apply(build_text, axis=1)


# ==========================
# Convert categories to numeric labels
# ==========================

categories = df["ISSUE CAT"].unique()

label_map = {cat: i for i, cat in enumerate(categories)}

df["label"] = df["ISSUE CAT"].map(label_map)


# Save label map
os.makedirs("model", exist_ok=True)

pd.Series(label_map).to_json("model/label_map.json")


# ==========================
# Create dataset
# ==========================

dataset = Dataset.from_pandas(df[["text", "label"]])


# ==========================
# Load base model
# ==========================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================
# Training arguments
# ==========================

args = SentenceTransformerTrainingArguments(

    output_dir="model/fine_tuned",

    num_train_epochs=4,

    per_device_train_batch_size=16,

    learning_rate=2e-5,

    warmup_ratio=0.1,

    fp16=False
)


# ==========================
# Loss function
# ==========================

loss = losses.BatchAllTripletLoss(model)


# ==========================
# Trainer
# ==========================

trainer = SentenceTransformerTrainer(

    model=model,

    args=args,

    train_dataset=dataset,

    loss=loss
)


# ==========================
# Train
# ==========================

trainer.train()


# ==========================
# Save model
# ==========================

model.save("model/fine_tuned")

print("Training complete. Model saved.")
