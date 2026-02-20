import asyncio
from database import db
from classifier import TicketClassifier

classifier = TicketClassifier()

async def process_batch(batch_id):
    tickets = await db.tickets.find({"batch_id": batch_id}).to_list(None)

    for t in tickets:
        cat, conf = classifier.classify(t["description"])

        await db.tickets.update_one(
            {"_id": t["_id"]},
            {"$set": {
                "category": cat,
                "confidence": conf,
                "classified": True
            }}
        )

    await db.batches.update_one(
        {"batch_id": batch_id},
        {"$set": {"status": "completed"}}
    )
