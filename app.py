
from predict_engine import classify_file
import os

INPUT = "data/incoming_tickets.xlsx"
OUTPUT = "output/classified_tickets.xlsx"

if __name__ == "__main__":
    if not os.path.exists(INPUT):
        print("Input file not found:", INPUT)
    else:
        classify_file(INPUT, OUTPUT)
