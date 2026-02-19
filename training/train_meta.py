import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_excel("data/meta_training_data.xlsx")

X = df.drop(columns=["true_label"])
y = df["true_label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

joblib.dump(model, "models/meta.pkl")

print("Meta model saved.")
