from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CATEGORIES = [
    "IT - System linkage issue",
    "IT - System Access issue",
    "IT – System Version issue",
    "IT – Data entry handholding",
    "IT – Master Data/ mapping issue",
    "User - Mapping missing",
    "User – Master data delayed input",
    "User - Logic changes during ABP",
    "User – Master data incorporation in system",
    "User – System Knowledge Gap",
    "User - Logic mistakes in excel vs system",
    "User - Multiple versions issue in excel"
]

class TicketClassifier:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.category_embeddings = self.model.encode(CATEGORIES)

    def classify(self, text):
        embedding = self.model.encode([text])
        similarity = cosine_similarity(embedding, self.category_embeddings)
        idx = np.argmax(similarity)
        confidence = float(similarity[0][idx])
        return CATEGORIES[idx], confidence
