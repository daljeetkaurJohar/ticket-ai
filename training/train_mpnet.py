from sentence_transformers import SentenceTransformer
import json, os

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

with open("data/category_examples.json", "r") as f:
    data = json.load(f)

embeddings_store = {}

for category, examples in data.items():
    embeddings = model.encode(examples, convert_to_numpy=True)
    embeddings_store[category] = embeddings.tolist()

os.makedirs("models/mpnet", exist_ok=True)

with open("models/mpnet/embeddings.json", "w") as f:
    json.dump(embeddings_store, f)

print("MPNet embeddings saved.")
