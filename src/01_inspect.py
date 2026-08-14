import pandas as pd

files = [
    "data/raw/sram_1.csv",
    "data/raw/sram_2.csv",
    "data/raw/sram_3.csv"
]

for file in files:

    print("\n" + "=" * 50)
    print(file)
    print("=" * 50)

    df = pd.read_csv(file, encoding="latin1", low_memory=False)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nFirst 3 Rows:")
    print(df.head(3))

print("\nInspection Completed")