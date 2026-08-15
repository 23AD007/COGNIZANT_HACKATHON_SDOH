from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


INPUT = Path("data/processed/member_sdoh_features.csv")
OUTPUT_DIR = Path("data/processed/sdoh_audit")


# Keep identifiers separate from model features.
IDENTIFIER_COLUMNS = {
    "member_id",
    "county_fips",
}

# These are demographic/member attributes, not SDOH features.
DEMOGRAPHIC_COLUMNS = {
    "birthdate",
    "age",
    "gender",
    "race",
    "ethnicity",
    "marital_status",
}

# Geography and source metadata should not enter the initial model.
METADATA_COLUMNS = {
    "city",
    "state",
    "zip",
    "county",
    "fips",
    "lat",
    "lon",
    "county_name",
    "state_abbr",
    "state_name",
    "places_year",
    "prevalence_type",
}

# These are source-level columns that are not themselves model features.
NON_FEATURE_COLUMNS = (
    IDENTIFIER_COLUMNS
    | DEMOGRAPHIC_COLUMNS
    | METADATA_COLUMNS
)


def numeric_summary(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []

    for col in feature_columns:
        s = pd.to_numeric(df[col], errors="coerce")

        rows.append(
            {
                "feature": col,
                "dtype": str(df[col].dtype),
                "numeric_values": int(s.notna().sum()),
                "missing_values": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean() * 100),
                "unique_values": int(s.nunique(dropna=True)),
                "zero_variance": bool(s.nunique(dropna=True) <= 1),
                "min": s.min(),
                "max": s.max(),
                "mean": s.mean(),
                "median": s.median(),
            }
        )

    return pd.DataFrame(rows)


def duplicate_feature_pairs(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    rows = []

    numeric = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    for i, col_a in enumerate(feature_columns):
        for col_b in feature_columns[i + 1:]:
            a = numeric[col_a]
            b = numeric[col_b]

            valid = a.notna() & b.notna()

            if valid.sum() < 2:
                continue

            correlation = a[valid].corr(b[valid])

            if pd.notna(correlation) and abs(correlation) >= 0.99:
                rows.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": float(correlation),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "feature_a",
            "feature_b",
            "correlation",
        ],
    )


def audit() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT)

    print("=" * 60)
    print("SDOH FEATURE QUALITY AUDIT")
    print("=" * 60)

    print(f"Input rows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # 1. Identify candidate model features
    # ---------------------------------------------------------

    feature_columns = [
        col
        for col in df.columns
        if col not in NON_FEATURE_COLUMNS
    ]

    print(f"Candidate feature columns: {len(feature_columns)}")

    # ---------------------------------------------------------
    # 2. Numeric / non-numeric inspection
    # ---------------------------------------------------------

    numeric_columns = []
    non_numeric_columns = []

    for col in feature_columns:
        converted = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        original_non_missing = df[col].notna().sum()
        converted_non_missing = converted.notna().sum()

        if (
            original_non_missing == 0
            or converted_non_missing == original_non_missing
        ):
            numeric_columns.append(col)
        else:
            non_numeric_columns.append(col)

    print("\nNumeric features:")
    for col in numeric_columns:
        print(f"  {col}")

    print("\nNon-numeric/problematic features:")
    for col in non_numeric_columns:
        print(f"  {col}")

    # ---------------------------------------------------------
    # 3. Missingness + ranges + variance
    # ---------------------------------------------------------

    summary = numeric_summary(
        df,
        feature_columns,
    )

    summary.to_csv(
        OUTPUT_DIR / "feature_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Constant / zero-variance features
    # ---------------------------------------------------------

    constant_features = summary.loc[
        summary["zero_variance"],
        "feature",
    ].tolist()

    pd.DataFrame(
        {"feature": constant_features}
    ).to_csv(
        OUTPUT_DIR / "constant_features.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Highly duplicated / correlated features
    # ---------------------------------------------------------

    correlation_pairs = duplicate_feature_pairs(
        df,
        numeric_columns,
    )

    correlation_pairs.to_csv(
        OUTPUT_DIR / "high_correlation_pairs.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Non-numeric values
    # ---------------------------------------------------------

    non_numeric_report = []

    for col in non_numeric_columns:
        s = df[col]

        non_numeric_report.append(
            {
                "feature": col,
                "dtype": str(s.dtype),
                "non_missing": int(s.notna().sum()),
                "unique_values": int(s.nunique(dropna=True)),
                "sample_values": " | ".join(
                    s.dropna()
                    .astype(str)
                    .drop_duplicates()
                    .head(10)
                    .tolist()
                ),
            }
        )

    pd.DataFrame(non_numeric_report).to_csv(
        OUTPUT_DIR / "non_numeric_features.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 7. Full missingness report
    # ---------------------------------------------------------

    missingness = (
        df[feature_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .rename("missing_count")
        .to_frame()
    )

    missingness["total_rows"] = len(df)

    missingness["missing_pct"] = (
        missingness["missing_count"]
        / len(df)
        * 100
    )

    missingness.to_csv(
        OUTPUT_DIR / "missingness.csv"
    )

    # ---------------------------------------------------------
    # 8. Save candidate feature list
    # ---------------------------------------------------------

    pd.DataFrame(
        {
            "feature": feature_columns,
            "numeric": [
                col in numeric_columns
                for col in feature_columns
            ],
        }
    ).to_csv(
        OUTPUT_DIR / "candidate_features.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 9. Print important findings
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("AUDIT RESULTS")
    print("=" * 60)

    print(f"Candidate features: {len(feature_columns)}")
    print(f"Numeric features: {len(numeric_columns)}")
    print(f"Non-numeric features: {len(non_numeric_columns)}")
    print(f"Zero-variance features: {len(constant_features)}")
    print(
        f"Highly correlated pairs (|r| >= 0.99): "
        f"{len(correlation_pairs)}"
    )

    print("\nTop missing features:")

    print(
        summary[
            [
                "feature",
                "missing_values",
                "missing_pct",
            ]
        ]
        .sort_values(
            "missing_pct",
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )

    print("\nZero-variance features:")

    if constant_features:
        for col in constant_features:
            print(f"  {col}")
    else:
        print("  None")

    print("\nAudit files written to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    audit()