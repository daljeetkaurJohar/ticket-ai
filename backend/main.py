from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
import pandas as pd
from uuid import uuid4
from io import BytesIO
from database import db
from worker import process_batch
import os

app = FastAPI()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        return {"error": "Only Excel allowed"}

    content = await file.read()
    df = pd.read_excel(BytesIO(content))

    batch_id = str(uuid4())

    await db.batches.insert_one({
        "batch_id": batch_id,
        "filename": file.filename,
        "status": "processing"
    })

    tickets = []
    for _, row in df.iterrows():
        tickets.append({
            "batch_id": batch_id,
            "description": str(row["Description"]),
            "classified": False
        })

    if tickets:
        await db.tickets.insert_many(tickets)

    return {"batch_id": batch_id}
@app.post("/classify/{batch_id}")
async def classify(batch_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_batch, batch_id)
    return {"message": "Classification started"}
