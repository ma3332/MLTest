import pandas as pd

# Create two sample DataFrames
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40]
})

df2 = pd.DataFrame({
    'ID': [2, 3, 5, 6],
    'City': ['New York', 'London', 'Paris', 'Tokyo'],
    'Occupation': ['Engineer', 'Artist', 'Doctor', 'Developer']
})

# Perform an inner join on the 'ID' column
merged_df = pd.merge(df1, df2, on='ID', how='left')

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)
print("\nMerged DataFrame (Inner Join on 'ID'):")
print(merged_df)