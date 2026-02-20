import asyncio
from database import db
from classifier import TicketClassifier

classifier = TicketClassifier()

async def process_batch(batch_id):
    tickets = await db.tickets.find({"batch_id": batch_id}).to_list(None)

    for ticket in tickets:
        category, confidence = classifier.classify(ticket["description"])

        await db.tickets.update_one(
            {"_id": ticket["_id"]},
            {"$set": {
                "category": category,
                "confidence": confidence,
                "classified": True
            }}
        )
