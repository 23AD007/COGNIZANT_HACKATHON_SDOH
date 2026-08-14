from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "GEO_ID",
    "NAME",
    "P2_001N",
    "P2_002N",
    "P2_003N",
    "P2_004N",
]


def preprocess_acs(input_path, output_path):

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    if df.empty:
        raise ValueError(
            "ACS file is empty."
        )

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing ACS columns: {missing}"
        )

    df = df[REQUIRED_COLUMNS].copy()

    # Preserve GEO_ID for later geographic mapping
    df["GEO_ID"] = (
        df["GEO_ID"]
        .astype("string")
        .str.strip()
    )

    df["NAME"] = (
        df["NAME"]
        .astype("string")
        .str.strip()
    )

    numeric_columns = [
        "P2_001N",
        "P2_002N",
        "P2_003N",
        "P2_004N",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.drop_duplicates(
        subset="GEO_ID"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Processed ACS rows: {len(df):,}"
    )

    print(
        f"Saved: {output_path}"
    )

    return df