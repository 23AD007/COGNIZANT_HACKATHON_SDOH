import pandas as pd
import os

files = {
    "sram_1": "data/raw/sram_1.csv",
    "sram_2": "data/raw/sram_2.csv",
    "sram_3": "data/raw/sram_3.csv"
}

os.makedirs("data/processed", exist_ok=True)

for name, file in files.items():

    print("\nProcessing:", name)

    df = pd.read_csv(
        file,
        encoding="latin1",
        low_memory=False
    )

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean CensusTract
    df["CensusTract20"] = (
        df["CensusTract20"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.zfill(11)
    )

    # Clean State and County
    for col in ["State", "County20", "County24"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Convert other columns to numeric
    for col in df.columns:
        if col not in ["CensusTract20", "State", "County20", "County24"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Check duplicate CensusTract
    print("Rows:", len(df))
    print("Duplicate CensusTract:", df["CensusTract20"].duplicated().sum())

    # Save
    output = f"data/processed/{name}_clean.csv"

    df.to_csv(
        output,
        index=False,
        encoding="utf-8"
    )

    print("Saved:", output)

print("\nPreprocessing Completed")