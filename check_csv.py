import pandas as pd

# Read the first few rows of the CSV
df = pd.read_csv('topical_chat.csv', nrows=5)

# Print column names
print("Column names:", df.columns.tolist())

# Print first few rows
print("\nFirst few rows:")
print(df.head()) 