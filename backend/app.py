from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Directory to upload files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.xlsx'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        return jsonify({"message": "File uploaded successfully", "filename": file.filename}), 200
    return jsonify({"error": "File type not supported"}), 400

@app.route('/categorize', methods=['POST'])
def categorize_tickets():
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({"error": "No filename provided"}), 400
    
    filename = data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404

    # Read Excel file
    df = pd.read_excel(filepath)

    # Dummy categorization logic (to be replaced with actual logic)
    df['category'] = df['ticket_name'].apply(lambda x: 'General' if 'issue' in x.lower() else 'Query')
    
    categorized_data = df.to_dict(orient='records')
    return jsonify(categorized_data), 200

if __name__ == '__main__':
    app.run(debug=True)