import pandas as pd

# Read cleaned files
sram1 = pd.read_csv("data/processed/sram_1_clean.csv")
sram2 = pd.read_csv("data/processed/sram_2_clean.csv")
sram3 = pd.read_csv("data/processed/sram_3_clean.csv")

# Remove repeated common columns from SRAM 2 and 3
sram2 = sram2.drop(columns=["State", "County20", "County24"])
sram3 = sram3.drop(columns=["State", "County20", "County24"])

# Merge using CensusTract20
df = sram1.merge(sram2, on="CensusTract20", how="inner")
df = df.merge(sram3, on="CensusTract20", how="inner")

print("Final Rows:", len(df))
print("Final Columns:", len(df.columns))

# Check key
print("Duplicate CensusTract20:", df["CensusTract20"].duplicated().sum())

# Save
df.to_csv("data/processed/sram_2025_merged.csv", index=False)

print("Merged dataset saved successfully!")