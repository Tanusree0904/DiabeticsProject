import pandas as pd
import os

print("Current folder:", os.getcwd())  # This shows where the file will be saved

data = pd.read_csv("data.csv")
data.to_excel("data.xlsx", index=False)

print("CSV has been converted to Excel!")