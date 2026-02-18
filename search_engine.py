
def build_full_context(row):
    return " | ".join([f"{col}: {row[col]}" for col in row.index])

def search_rows(df, query):
    query = query.lower()
    mask = df.apply(lambda r: query in build_full_context(r).lower(), axis=1)
    return df[mask]
