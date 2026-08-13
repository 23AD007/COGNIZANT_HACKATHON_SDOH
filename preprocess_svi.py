import pandas as pd
import os

# 1. Load the raw dataset
df = pd.read_csv("data/raw/SVI.csv")

print("Original Shape:")
print(df.shape)

# 2. Clean column names
df.columns = df.columns.str.strip()

# 3. Remove completely empty columns
df = df.dropna(axis=1, how="all")

# 4. Remove completely empty rows
df = df.dropna(axis=0, how="all")

# 5. Remove duplicate rows
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()

# 6. Remove the repeated header row
# The dataset contains "Geography" as the first GEO_ID value
df = df[df["GEO_ID"].astype(str).str.strip().str.lower() != "geography"]

# 7. Convert population columns to numeric
population_columns = [
    "P2_001N",
    "P2_002N",
    "P2_003N",
    "P2_004N"
]

for col in population_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 8. Remove rows missing important geographic information
df = df.dropna(subset=["GEO_ID", "NAME"])

# 9. Clean text columns
df["GEO_ID"] = df["GEO_ID"].astype(str).str.strip()
df["NAME"] = df["NAME"].astype(str).str.strip()

# 10. Reset row numbers
df = df.reset_index(drop=True)

# 11. Create processed folder
os.makedirs("data/processed", exist_ok=True)

# 12. Save cleaned dataset as CSV
df.to_csv(
    "data/processed/svi_preprocessed.csv",
    index=False
)

# 13. Save cleaned dataset as Excel
df.to_excel(
    "data/processed/svi_preprocessed.xlsx",
    index=False
)

# 14. Display final information
print("\n===== PREPROCESSING COMPLETED =====")

print("\nDuplicates removed:")
print(duplicates_before)

print("\nFinal Shape:")
print(df.shape)

print("\nFinal Columns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Cleaned Rows:")
print(df.head())

print("\nCleaned dataset saved successfully!")
print("CSV: data/processed/svi_preprocessed.csv")
print("Excel: data/processed/svi_preprocessed.xlsx")