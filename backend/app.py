# backend/app.py

from flask import Flask, request, jsonify
import pandas as pd
import os

from classifier import predict_ticket

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# ------------------------------------
# Health Check
# ------------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Backend running"}), 200


# ------------------------------------
# Upload Excel File
# ------------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith(".xlsx"):
        return jsonify({"error": "Only .xlsx files supported"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    return jsonify({
        "message": "File uploaded successfully",
        "filename": file.filename
    }), 200


# ------------------------------------
# Categorize Tickets
# ------------------------------------
@app.route("/categorize", methods=["POST"])
def categorize_tickets():

    data = request.json

    if not data or "filename" not in data:
        return jsonify({"error": "Filename required"}), 400

    filename = data["filename"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404

    # Read Excel file
    df = pd.read_excel(filepath)

    if df.empty:
        return jsonify({"error": "File is empty"}), 400

    predictions = []
    confidences = []
    sources = []

    for _, row in df.iterrows():

        # Combine relevant text fields safely
        text = " ".join([
            str(row.get("Issue", "")),
            str(row.get("Description", "")),
            str(row.get("Remarks", "")),
            str(row.get("Ticket Description", "")),
            str(row.get("Ticket Summary", "")),
            str(row.get("Ticket Details", ""))
        ])

        result = predict_ticket(text)

        predictions.append(result["category"])
        confidences.append(result["confidence"])
        sources.append(result["source"])

    df["Predicted Category"] = predictions
    df["Confidence"] = confidences
    df["Classification Source"] = sources

    # Save updated file
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"classified_{filename}")
    df.to_excel(output_path, index=False)

    return jsonify({
        "message": "Classification completed",
        "output_file": f"classified_{filename}",
        "total_records": len(df)
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
