import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_excel("labeled_tickets.xlsx")

# Combine text columns if needed
df["text"] = df["Ticket Summary"].fillna("") + " " + df["Ticket Details"].fillna("")

# Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["Category"])

num_labels = len(label_encoder.classes_)

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])

# Load tokenizer & model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

training_args = TrainingArguments(
    output_dir="./model_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Save model
trainer.save_model("./ticket_classifier_model")
tokenizer.save_pretrained("./ticket_classifier_model")

print("Model training complete.")
