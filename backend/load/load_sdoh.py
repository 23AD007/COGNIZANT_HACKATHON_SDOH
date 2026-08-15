import pandas as pd
from sqlalchemy import text

from backend.database import engine


COUNTY_FILE = "data/processed/county_sdoh_features.csv"
MEMBER_FILE = "data/processed/member_sdoh_features.csv"
CLINICAL_FILE = "data/processed/clinical_patient_coverage.csv"

SCHEMA = "sdoh"


def load_csv(file_path, table_name):
    print(f"\nLoading {file_path}...")

    df = pd.read_csv(file_path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    df.to_sql(
        table_name,
        con=engine,
        schema=SCHEMA,
        if_exists="replace",
        index=False,
        method="multi"
    )

    print(f"Successfully loaded {len(df)} rows into {SCHEMA}.{table_name}")


def main():

    print("=" * 60)
    print("UC09 SDOH DATABASE LOADER")
    print("=" * 60)

    # Make sure schema exists
    with engine.begin() as connection:
        connection.execute(
            text("CREATE SCHEMA IF NOT EXISTS sdoh")
        )

    # Load datasets
    load_csv(
        COUNTY_FILE,
        "county_sdoh_features"
    )

    load_csv(
        MEMBER_FILE,
        "member_sdoh_features"
    )

    load_csv(
        CLINICAL_FILE,
        "clinical_patient_coverage"
    )

    print("\n" + "=" * 60)
    print("ALL DATASETS LOADED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()