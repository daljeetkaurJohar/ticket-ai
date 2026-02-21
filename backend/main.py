from fastapi import FastAPI, UploadFile
import pandas as pd
from classifier import categorize_ticket

app = FastAPI()

@app.post("/categorize/")
async def categorize(file: UploadFile):

    df = pd.read_excel(file.file)

    results = []

    for _, row in df.iterrows():
        category = categorize_ticket(
            row.get("Short Description", ""),
            row.get("Description", "")
        )
        results.append(category)

    df["Predicted Issue Category"] = results

    output_path = "categorized_output.xlsx"
    df.to_excel(output_path, index=False)

    return {"message": "Categorization completed"}
