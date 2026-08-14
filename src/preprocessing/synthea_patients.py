from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "id",
    "birthdate",
    "deathdate",
    "marital",
    "race",
    "ethnicity",
    "gender",
    "city",
    "state",
    "county",
    "fips",
    "zip",
    "lat",
    "lon",
    "healthcare_expenses",
    "healthcare_coverage",
    "income",
]


FINAL_COLUMNS = [
    "member_id",
    "birthdate",
    "deathdate",
    "age",
    "marital",
    "race",
    "ethnicity",
    "gender",
    "city",
    "state",
    "county",
    "fips",
    "zip",
    "lat",
    "lon",
    "healthcare_expenses",
    "healthcare_coverage",
    "income",
]


def preprocess_patients(input_path, output_path):

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Patient file not found: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    print(f"Raw patient shape: {df.shape}")

    if df.empty:
        raise ValueError(
            "patients.csv contains zero records."
        )

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing patient columns: {missing}"
        )

    # Keep only required non-PII fields
    df = df[REQUIRED_COLUMNS].copy()

    # Rename Synthea ID
    df.rename(
        columns={"id": "member_id"},
        inplace=True
    )

    # --------------------------------------------------
    # Member ID
    # --------------------------------------------------

    df["member_id"] = (
        df["member_id"]
        .astype("string")
        .str.strip()
    )

    df = df[
        df["member_id"].notna()
        & (df["member_id"] != "")
    ].copy()

    # --------------------------------------------------
    # Dates
    # --------------------------------------------------

    df["birthdate"] = pd.to_datetime(
        df["birthdate"],
        errors="coerce"
    )

    df["deathdate"] = pd.to_datetime(
        df["deathdate"],
        errors="coerce"
    )

    # --------------------------------------------------
    # Age
    # --------------------------------------------------

    today = pd.Timestamp.today().normalize()

    df["age"] = (
        (today - df["birthdate"]).dt.days
        / 365.25
    ).round()

    df["age"] = df["age"].astype("Int64")

    # --------------------------------------------------
    # FIPS
    # --------------------------------------------------

    df["fips"] = (
        df["fips"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    # --------------------------------------------------
    # ZIP
    # --------------------------------------------------

    df["zip"] = (
        df["zip"]
        .astype("string")
        .str.extract(
            r"(\d{5})",
            expand=False
        )
    )

    # --------------------------------------------------
    # Numeric fields
    # --------------------------------------------------

    numeric_columns = [
        "lat",
        "lon",
        "healthcare_expenses",
        "healthcare_coverage",
        "income",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------
    # Categorical fields
    # --------------------------------------------------

    categorical_columns = [
        "marital",
        "race",
        "ethnicity",
        "gender",
        "city",
        "state",
        "county",
    ]

    for col in categorical_columns:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
        )

    # --------------------------------------------------
    # Duplicate members
    # --------------------------------------------------

    duplicate_count = df["member_id"].duplicated().sum()

    if duplicate_count:
        print(
            f"Removing {duplicate_count} duplicate members."
        )

        df = df.drop_duplicates(
            subset="member_id",
            keep="first"
        )

    # --------------------------------------------------
    # Final ordering
    # --------------------------------------------------

    df = df[FINAL_COLUMNS]

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    if df.empty:
        raise ValueError(
            "No members remain after preprocessing."
        )

    if df["member_id"].isna().any():
        raise ValueError(
            "member_id contains null values."
        )

    if not df["member_id"].is_unique:
        raise ValueError(
            "member_id is not unique."
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Processed patients: {len(df):,}"
    )

    print(
        f"Saved: {output_path}"
    )

    return df