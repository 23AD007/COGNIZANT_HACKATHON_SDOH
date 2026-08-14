from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "CensusTract20",
    "State",
    "County20",
    "County24",
    "Urban",
    "POP2020",
    "OHU2020",
    "LowIncomeTracts",
    "PovertyRate",
    "MedianFamilyIncome",
    "TractLOWI",
    "TractKids",
    "TractSeniors",
    "TractHUNV",
    "TractSNAP",
    "TractVeteran",
    "TractTribalArea",
]


SRAM_FEATURE_COLUMNS = [
    "DD_SRAM_LILATracts_1And10",
    "DD_SRAM_LILATracts_halfAnd10",
    "DD_SRAM_LILATracts_1And20",
    "DD_SRAM_LILATracts_Vehicle",
    "DD_SRAM_HUNVFlag",
    "DD_SRAM_LAPOP1_10",
    "DD_SRAM_LAPOP05_10",
    "DD_SRAM_LAPOP1_20",
    "DD_SRAM_LALOWI1_10",
    "DD_SRAM_LALOWI05_10",
    "DD_SRAM_LALOWI1_20",
    "SD_SRAM_LILATracts_1And10",
    "SD_SRAM_LILATracts_halfAnd10",
    "SD_SRAM_LILATracts_1And20",
    "SD_SRAM_LILATracts_Vehicle",
    "SD_SRAM_HUNVFlag",
    "SD_SRAM_LAPOP1_10",
    "SD_SRAM_LAPOP05_10",
    "SD_SRAM_LAPOP1_20",
    "SD_SRAM_LALOWI1_10",
    "SD_SRAM_LALOWI05_10",
    "SD_SRAM_LALOWI1_20",
]


def preprocess_sram(input_path, output_path):

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"SRAM file not found: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        low_memory=False
    )

    if df.empty:
        raise ValueError(
            "SRAM file is empty."
        )

    required = (
        REQUIRED_COLUMNS
        + SRAM_FEATURE_COLUMNS
    )

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing SRAM columns:\n"
            + "\n".join(missing)
        )

    df = df[required].copy()

    # --------------------------------------------------
    # Geographic key
    # --------------------------------------------------

    df["CensusTract20"] = (
        df["CensusTract20"]
        .astype("string")
        .str.strip()
        .str.replace(
            r"\.0$",
            "",
            regex=True
        )
    )

    if df["CensusTract20"].isna().any():
        raise ValueError(
            "SRAM contains missing CensusTract20 values."
        )

    if not df["CensusTract20"].is_unique:
        duplicates = (
            df["CensusTract20"]
            .duplicated()
            .sum()
        )

        print(
            f"Removing {duplicates} duplicate "
            "CensusTract20 records."
        )

        df = df.drop_duplicates(
            subset="CensusTract20",
            keep="first"
        )

    # --------------------------------------------------
    # Numeric fields
    # --------------------------------------------------

    numeric_columns = [
        c for c in df.columns
        if c != "CensusTract20"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------
    # Basic range validation
    # --------------------------------------------------

    rate_columns = [
        "PovertyRate"
    ]

    for col in rate_columns:

        if col in df.columns:

            invalid = (
                df[col].notna()
                & (
                    (df[col] < 0)
                    | (df[col] > 100)
                )
            )

            if invalid.any():
                raise ValueError(
                    f"{col} contains values "
                    "outside 0-100."
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
        f"Processed SRAM tracts: "
        f"{len(df):,}"
    )

    print(
        f"Saved: {output_path}"
    )

    return df