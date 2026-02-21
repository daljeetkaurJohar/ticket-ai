import pandas as pd

def load_data(file_path):
    """Load data from an Excel file."""
    return pd.read_excel(file_path)

def categorize_tickets(df):
    """Dynamically categorize tickets based on the content of columns."""
    categories = []

    for index, row in df.iterrows():
        issue = row['Issue']
        user = row['Users']
        description = row['Description']

        # Example logic for categorization
        if "error" in description.lower():
            categories.append("Error")
        elif "request" in description.lower():
            categories.append("Request")
        else:
            categories.append("General Inquiry")

    df['Category'] = categories
    return df

def main():
    file_path = 'path_to_your_excel_file.xlsx'  # Update this path accordingly
    df = load_data(file_path)
    categorized_df = categorize_tickets(df)
    print(categorized_df)

if __name__ == "__main__":
    main()