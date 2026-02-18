
import pandas as pd
import matplotlib.pyplot as plt

FILE = "output/classified_tickets.xlsx"

df = pd.read_excel(FILE)

counts = df["Predicted Category"].value_counts()

counts.plot(kind="bar")
plt.title("Category Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
