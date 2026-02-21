from flask import Flask, request
import pandas as pd

app = Flask(__name__)

@app.route('/categorize', methods=['POST'])
def categorize_ticket():
    
    # Check if an Excel file is uploaded
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400
    file = request.files['file']
    
    if file.filename == '':
        return {'error': 'No selected file'}, 400
        
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file)
    
    # Assuming the Excel has a column called 'ticket_description'
    categories = []
    for description in df['ticket_description']:
        # Implement your categorization logic here
        # Placeholder logic for categorization
        category = 'General' if 'issue' in description.lower() else 'Inquiry'
        categories.append(category)
    
    df['category'] = categories
    
    # Return the categorized tickets as JSON
    result = df.to_dict(orient='records')
    return {'categorized_tickets': result}, 200

if __name__ == '__main__':
    app.run(debug=True)