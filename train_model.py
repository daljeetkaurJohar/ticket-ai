import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import numpy as np
import evaluate
import os

# -----------------------------
# Load Data
# -----------------------------
xls = pd.ExcelFile("data/issue_category.xlsx")

df_list = []
for sheet in xls.sheet_names:
    temp = pd.read_excel(xls, sheet)
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

# Combine text
df["text"] = (
    df["Short Description"].astype(str) + " " +
    df["Description"].astype(str)
)

df = df[["text", "Issue"]].dropna()

# -----------------------------
# Encode Labels
# -----------------------------
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Issue"])

num_labels = len(label_encoder.classes_)

# Save label encoder
import joblib
joblib.dump(label_encoder, "label_encoder.pkl")

# -----------------------------
# Train Test Split
# -----------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------
# Tokenization
# -----------------------------
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=256
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=256
)

train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": val_labels
})

# -----------------------------
# Model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# -----------------------------
# Metrics
# -----------------------------
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# Save final model
trainer.save_model("./model")
tokenizer.save_pretrained("./model")

print("Training complete.")
