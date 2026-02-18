
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json


class FineTunedClassifier:

    def __init__(self):

        self.model = AutoModelForSequenceClassification.from_pretrained("model")

        self.tokenizer = AutoTokenizer.from_pretrained("model")

        with open("model/label_map.json") as f:
            self.label_map = json.load(f)

        self.reverse_map = {v: k for k, v in self.label_map.items()}


    def predict(self, row):

        text = " | ".join([f"{col}: {row[col]}" for col in row.index])

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)

        outputs = self.model(**inputs)

        pred = torch.argmax(outputs.logits, dim=1).item()

        confidence = torch.softmax(outputs.logits, dim=1).max().item()

        return self.reverse_map[pred], confidence

