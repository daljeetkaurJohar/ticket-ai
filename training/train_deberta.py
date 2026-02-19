import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json, os

MODEL_NAME = "microsoft/deberta-v3-base"

df = pd.read_excel("data/incoming_tickets.xlsx")
df = df.dropna(subset=["ISSUE CAT"])

def build_text(row):
    cols = ["Ticket Summary", "Ticket Details", "Work Notes", "Solution"]
    return " | ".join([str(row[c]) for c in cols if c in row and pd.notna(row[c])])

df["text"] = df.apply(build_text, axis=1)

labels = df["ISSUE CAT"].unique()
label_map = {l: i for i, l in enumerate(labels)}
df["label"] = df["ISSUE CAT"].map(label_map)

dataset = Dataset.from_pandas(df[["text", "label"]])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels)
)

training_args = TrainingArguments(
    output_dir="models/deberta",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("models/deberta")
tokenizer.save_pretrained("models/deberta")

with open("models/deberta/label_map.json", "w") as f:
    json.dump(label_map, f)

print("DeBERTa trained.")
