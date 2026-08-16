"""
COUNTY RISK GENERATION

Purpose
-------
Generate county-level SDOH/member risk scores from the validated
member_risk_scores.csv output.

Important rules
---------------
1. Never manually assign missing county_fips.
2. Members with county_fips = NaN remain unresolved.
3. County risk is calculated only from members with valid county_fips.
4. Member risk probabilities are aggregated to county-level metrics.
5. County-level SDOH features are summarized where available.
6. No target leakage is introduced.
7. Output is deterministic and reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

MEMBER_RISK_FILE = PROCESSED_DIR / "member_risk_scores.csv"
MEMBER_MODEL_FILE = PROCESSED_DIR / "member_model_features.csv"

COUNTY_RISK_FILE = PROCESSED_DIR / "county_risk_scores.csv"
COUNTY_VALIDATION_FILE = (
    PROCESSED_DIR / "county_risk_validation.csv"
)
COUNTY_MEMBER_COVERAGE_FILE = (
    PROCESSED_DIR / "county_member_coverage.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MEMBER_ID_COLUMN = "member_id"
PATIENT_ID_COLUMN = "patient_id"
COUNTY_COLUMN = "county_fips"

RISK_COLUMN = "risk_probability"
RISK_PERCENTILE_COLUMN = "risk_percentile"
RISK_BAND_COLUMN = "risk_band"

TARGET_COLUMN = "target_inpatient_any"

VALID_RISK_BANDS = {
    "Very Low",
    "Low",
    "Medium",
    "High",
    "Very High",
}

RANDOM_STATE = 42


# ============================================================
# LOGGING
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_county(value):
    """
    Normalize county FIPS values.

    Examples
    --------
    25017.0 -> "25017"
    "25017" -> "25017"
    "025017" -> "25017"
    NaN -> NaN
    """

    if pd.isna(value):
        return np.nan

    text = str(value).strip()

    if text == "":
        return np.nan

    try:
        number = float(text)

        if not np.isfinite(number):
            return np.nan

        return str(int(number)).zfill(5)

    except (ValueError, TypeError):
        digits = "".join(ch for ch in text if ch.isdigit())

        if not digits:
            return np.nan

        return digits.zfill(5)[-5:]


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def percentile_rank(series: pd.Series) -> pd.Series:
    """
    Percentile rank from 0 to 100.
    """

    numeric = safe_numeric(series)

    if numeric.notna().sum() == 0:
        return pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

    return numeric.rank(
        method="average",
        pct=True,
    ) * 100.0


# ============================================================
# LOAD MEMBER RISK
# ============================================================

def load_member_risk() -> pd.DataFrame:

    section("LOADING MEMBER RISK")

    if not MEMBER_RISK_FILE.exists():
        raise FileNotFoundError(
            f"""
Member risk file not found:

{MEMBER_RISK_FILE}

Run:

py -3.12 -m src.modeling.member_risk
"""
        )

    df = pd.read_csv(MEMBER_RISK_FILE)

    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# VALIDATE MEMBER RISK
# ============================================================

def validate_member_risk(df: pd.DataFrame) -> None:

    section("VALIDATING MEMBER RISK DATA")

    required = {
        MEMBER_ID_COLUMN,
        COUNTY_COLUMN,
        RISK_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Required columns missing from member risk data: "
            + ", ".join(sorted(missing))
        )

    if df[MEMBER_ID_COLUMN].duplicated().any():
        duplicate_count = int(
            df[MEMBER_ID_COLUMN].duplicated().sum()
        )

        raise ValueError(
            f"{duplicate_count} duplicate member_id values found."
        )

    df[COUNTY_COLUMN] = df[COUNTY_COLUMN].apply(
        normalize_county
    )

    df[RISK_COLUMN] = safe_numeric(
        df[RISK_COLUMN]
    )

    missing_risk = int(
        df[RISK_COLUMN].isna().sum()
    )

    if missing_risk:
        raise ValueError(
            f"{missing_risk} members have missing risk_probability."
        )

    invalid_risk = (
        (df[RISK_COLUMN] < 0)
        | (df[RISK_COLUMN] > 1)
    )

    if invalid_risk.any():
        raise ValueError(
            "risk_probability must be between 0 and 1."
        )

    duplicate_counties = (
        df[COUNTY_COLUMN].isna().sum()
    )

    print(
        f"Members:              {len(df)}"
    )

    print(
        f"Members with county:  "
        f"{df[COUNTY_COLUMN].notna().sum()}"
    )

    print(
        f"Members without county:"
        f" {duplicate_counties}"
    )

    print(
        f"Risk min:             "
        f"{df[RISK_COLUMN].min():.4f}"
    )

    print(
        f"Risk max:             "
        f"{df[RISK_COLUMN].max():.4f}"
    )

    print(
        f"Risk mean:            "
        f"{df[RISK_COLUMN].mean():.4f}"
    )

    print()
    print(
        "County validation: PASSED"
    )


# ============================================================
# LOAD MEMBER MODEL FEATURES
# ============================================================

def load_member_model_features() -> pd.DataFrame | None:

    section("LOADING MEMBER MODEL FEATURES")

    if not MEMBER_MODEL_FILE.exists():

        print(
            "Member model feature file not found."
        )

        print(
            "County risk will be generated using "
            "member risk scores only."
        )

        return None

    df = pd.read_csv(
        MEMBER_MODEL_FILE
    )

    print(
        f"Rows:    {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    if MEMBER_ID_COLUMN not in df.columns:
        print(
            "WARNING: member_id missing from "
            "member model feature file."
        )

        return None

    return df


# ============================================================
# MERGE MODEL FEATURES
# ============================================================

def merge_model_features(
    member_risk: pd.DataFrame,
    model_features: pd.DataFrame | None,
) -> pd.DataFrame:

    if model_features is None:
        return member_risk.copy()

    feature_columns = [
        c
        for c in model_features.columns
        if c != MEMBER_ID_COLUMN
        and c != COUNTY_COLUMN
        and c != TARGET_COLUMN
    ]

    feature_columns = [
        c
        for c in feature_columns
        if c not in member_risk.columns
    ]

    if not feature_columns:
        return member_risk.copy()

    features = model_features[
        [MEMBER_ID_COLUMN] + feature_columns
    ].copy()

    features = features.drop_duplicates(
        subset=[MEMBER_ID_COLUMN]
    )

    merged = member_risk.merge(
        features,
        on=MEMBER_ID_COLUMN,
        how="left",
        validate="one_to_one",
        suffixes=("", "_model"),
    )

    return merged


# ============================================================
# COUNTY AGGREGATION
# ============================================================

def aggregate_county_risk(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("AGGREGATING MEMBER RISK TO COUNTY")

    working = df.copy()

    working = working[
        working[COUNTY_COLUMN].notna()
    ].copy()

    if working.empty:
        raise ValueError(
            "No members have a valid county_fips."
        )

    print(
        f"Members used for county aggregation: "
        f"{len(working)}"
    )

    print(
        f"Counties represented: "
        f"{working[COUNTY_COLUMN].nunique()}"
    )

    # --------------------------------------------------------
    # Basic risk aggregation
    # --------------------------------------------------------

    county = (
        working
        .groupby(COUNTY_COLUMN)
        .agg(
            member_count=(
                MEMBER_ID_COLUMN,
                "count",
            ),

            mean_member_risk=(
                RISK_COLUMN,
                "mean",
            ),

            median_member_risk=(
                RISK_COLUMN,
                "median",
            ),

            max_member_risk=(
                RISK_COLUMN,
                "max",
            ),

            min_member_risk=(
                RISK_COLUMN,
                "min",
            ),

            std_member_risk=(
                RISK_COLUMN,
                "std",
            ),
        )
        .reset_index()
    )

    county["std_member_risk"] = (
        county["std_member_risk"]
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # High-risk population
    # --------------------------------------------------------

    high_risk = (
        working[RISK_COLUMN] >= 0.50
    )

    very_high_risk = (
        working[RISK_COLUMN] >= 0.75
    )

    high_risk_df = working.copy()

    high_risk_df["high_risk_flag"] = (
        high_risk.astype(int)
    )

    high_risk_df["very_high_risk_flag"] = (
        very_high_risk.astype(int)
    )

    high_counts = (
        high_risk_df
        .groupby(COUNTY_COLUMN)
        .agg(
            high_risk_member_count=(
                "high_risk_flag",
                "sum",
            ),

            very_high_risk_member_count=(
                "very_high_risk_flag",
                "sum",
            ),
        )
        .reset_index()
    )

    county = county.merge(
        high_counts,
        on=COUNTY_COLUMN,
        how="left",
        validate="one_to_one",
    )

    county[
        "high_risk_member_pct"
    ] = (
        county["high_risk_member_count"]
        / county["member_count"]
        * 100.0
    )

    county[
        "very_high_risk_member_pct"
    ] = (
        county["very_high_risk_member_count"]
        / county["member_count"]
        * 100.0
    )

    # --------------------------------------------------------
    # Mean risk as primary county risk
    # --------------------------------------------------------

    county[
        "county_risk_probability"
    ] = county["mean_member_risk"]

    county[
        "county_risk_percentile"
    ] = percentile_rank(
        county["county_risk_probability"]
    )

    return county


# ============================================================
# COUNTY SDOH AGGREGATION
# ============================================================

def add_sdoh_aggregates(
    county: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:

    section("AGGREGATING COUNTY SDOH FEATURES")

    # We only aggregate numeric features.
    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    excluded = {
        RISK_COLUMN,
        TARGET_COLUMN,
    }

    # Do not aggregate identifiers.
    numeric_columns = [
        c
        for c in numeric_columns
        if c not in excluded
        and c != COUNTY_COLUMN
        and c != "risk_percentile"
    ]

    # Remove obvious IDs / geographic identifiers.
    numeric_columns = [
        c
        for c in numeric_columns
        if c.lower()
        not in {
            "state_fips",
            "state_fips_places",
            "county_geoid",
            "county_geoid_places",
            "fips",
            "zip",
        }
    ]

    if not numeric_columns:
        print(
            "No numeric SDOH features available."
        )
        return county

    # Only use rows with valid county.
    working = df[
        df[COUNTY_COLUMN].notna()
    ].copy()

    # Convert to numeric.
    for column in numeric_columns:
        working[column] = safe_numeric(
            working[column]
        )

    # We do not want to create hundreds of unnecessary
    # duplicated features. Aggregate selected SDOH variables
    # using their member-level mean.
    sdoh = (
        working
        .groupby(COUNTY_COLUMN)[numeric_columns]
        .mean()
        .reset_index()
    )

    rename_map = {
        column: f"county_mean_{column}"
        for column in numeric_columns
    }

    sdoh = sdoh.rename(
        columns=rename_map
    )

    county = county.merge(
        sdoh,
        on=COUNTY_COLUMN,
        how="left",
        validate="one_to_one",
    )

    print(
        f"County SDOH aggregates added: "
        f"{len(numeric_columns)}"
    )

    return county


# ============================================================
# COUNTY RISK BAND
# ============================================================

def assign_risk_bands(
    county: pd.DataFrame,
) -> pd.DataFrame:

    section("ASSIGNING COUNTY RISK BANDS")

    percentile = county[
        "county_risk_percentile"
    ]

    conditions = [
        percentile <= 20,
        percentile <= 40,
        percentile <= 60,
        percentile <= 80,
        percentile > 80,
    ]

    choices = [
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High",
    ]

    county["risk_band"] = np.select(
        conditions,
        choices,
        default="Medium",
    )

    return county


# ============================================================
# COUNTY RANK
# ============================================================

def assign_county_rank(
    county: pd.DataFrame,
) -> pd.DataFrame:

    county = county.sort_values(
        [
            "county_risk_probability",
            "high_risk_member_pct",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)

    county["risk_rank"] = (
        np.arange(len(county)) + 1
    )

    return county


# ============================================================
# VALIDATE COUNTY OUTPUT
# ============================================================

def validate_county_output(
    county: pd.DataFrame,
) -> None:

    section("VALIDATING COUNTY RISK OUTPUT")

    required = {
        COUNTY_COLUMN,
        "member_count",
        "mean_member_risk",
        "county_risk_probability",
        "county_risk_percentile",
        "risk_band",
        "risk_rank",
    }

    missing = required - set(
        county.columns
    )

    if missing:
        raise ValueError(
            "County output missing columns: "
            + ", ".join(sorted(missing))
        )

    if county[COUNTY_COLUMN].isna().any():
        raise ValueError(
            "County output contains missing county_fips."
        )

    if county[COUNTY_COLUMN].duplicated().any():
        raise ValueError(
            "County output contains duplicate counties."
        )

    if county["member_count"].le(0).any():
        raise ValueError(
            "County member_count contains invalid values."
        )

    risk = safe_numeric(
        county["county_risk_probability"]
    )

    if risk.isna().any():
        raise ValueError(
            "County risk contains missing probabilities."
        )

    if ((risk < 0) | (risk > 1)).any():
        raise ValueError(
            "County risk probability must be between 0 and 1."
        )

    invalid_bands = (
        set(county["risk_band"].dropna().unique())
        - VALID_RISK_BANDS
    )

    if invalid_bands:
        raise ValueError(
            "Invalid county risk bands: "
            + ", ".join(sorted(invalid_bands))
        )

    if county["risk_rank"].duplicated().any():
        raise ValueError(
            "County risk ranks are not unique."
        )

    print(
        f"Counties:             {len(county)}"
    )

    print(
        f"Total members:        "
        f"{county['member_count'].sum()}"
    )

    print(
        f"Minimum county risk:  "
        f"{risk.min():.4f}"
    )

    print(
        f"Maximum county risk:  "
        f"{risk.max():.4f}"
    )

    print(
        f"Mean county risk:     "
        f"{risk.mean():.4f}"
    )

    print()
    print("Risk bands:")

    print(
        county["risk_band"]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "County risk validation: PASSED"
    )


# ============================================================
# MEMBER COVERAGE REPORT
# ============================================================

def create_member_coverage_report(
    member_risk: pd.DataFrame,
) -> pd.DataFrame:

    working = member_risk.copy()

    working["county_status"] = np.where(
        working[COUNTY_COLUMN].notna(),
        "Resolved",
        "Unresolved",
    )

    report = (
        working
        .groupby("county_status")
        .agg(
            member_count=(
                MEMBER_ID_COLUMN,
                "count",
            ),

            mean_member_risk=(
                RISK_COLUMN,
                "mean",
            ),
        )
        .reset_index()
    )

    total_members = len(working)

    report["member_pct"] = (
        report["member_count"]
        / total_members
        * 100.0
    )

    return report


# ============================================================
# VALIDATION REPORT
# ============================================================

def create_validation_report(
    member_risk: pd.DataFrame,
    county: pd.DataFrame,
) -> pd.DataFrame:

    total_members = len(member_risk)

    resolved_members = int(
        member_risk[COUNTY_COLUMN]
        .notna()
        .sum()
    )

    unresolved_members = (
        total_members
        - resolved_members
    )

    total_counties = len(county)

    rows = [
        {
            "metric": "total_members",
            "value": total_members,
        },
        {
            "metric": "members_with_county",
            "value": resolved_members,
        },
        {
            "metric": "members_without_county",
            "value": unresolved_members,
        },
        {
            "metric": "county_resolution_pct",
            "value": (
                resolved_members
                / total_members
                * 100.0
            ),
        },
        {
            "metric": "counties_generated",
            "value": total_counties,
        },
        {
            "metric": "total_members_used_for_county_risk",
            "value": int(
                county["member_count"].sum()
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# TOP COUNTIES
# ============================================================

def print_top_counties(
    county: pd.DataFrame,
    n: int = 10,
) -> None:

    section(
        f"TOP {n} HIGHEST-RISK COUNTIES"
    )

    columns = [
        COUNTY_COLUMN,
        "risk_rank",
        "member_count",
        "county_risk_probability",
        "county_risk_percentile",
        "high_risk_member_count",
        "high_risk_member_pct",
        "risk_band",
    ]

    available = [
        c for c in columns
        if c in county.columns
    ]

    print(
        county
        .sort_values("risk_rank")
        .head(n)[available]
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section("COUNTY RISK GENERATION")

    # --------------------------------------------------------
    # Load member risk
    # --------------------------------------------------------

    member_risk = load_member_risk()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_member_risk(
        member_risk
    )

    # --------------------------------------------------------
    # Load model features
    # --------------------------------------------------------

    model_features = (
        load_member_model_features()
    )

    # --------------------------------------------------------
    # Merge features
    # --------------------------------------------------------

    working = merge_model_features(
        member_risk,
        model_features,
    )

    # --------------------------------------------------------
    # Re-normalize county
    # --------------------------------------------------------

    working[COUNTY_COLUMN] = (
        working[COUNTY_COLUMN]
        .apply(normalize_county)
    )

    # --------------------------------------------------------
    # Aggregate member risk
    # --------------------------------------------------------

    county = aggregate_county_risk(
        working
    )

    # --------------------------------------------------------
    # Add SDOH aggregates
    # --------------------------------------------------------

    county = add_sdoh_aggregates(
        county,
        working,
    )

    # --------------------------------------------------------
    # Risk bands
    # --------------------------------------------------------

    county = assign_risk_bands(
        county
    )

    # --------------------------------------------------------
    # Rank counties
    # --------------------------------------------------------

    county = assign_county_rank(
        county
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_county_output(
        county
    )

    # --------------------------------------------------------
    # Coverage report
    # --------------------------------------------------------

    coverage = create_member_coverage_report(
        member_risk
    )

    validation = create_validation_report(
        member_risk,
        county,
    )

    # --------------------------------------------------------
    # Save county risk
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    county.to_csv(
        COUNTY_RISK_FILE,
        index=False,
    )

    validation.to_csv(
        COUNTY_VALIDATION_FILE,
        index=False,
    )

    coverage.to_csv(
        COUNTY_MEMBER_COVERAGE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print top counties
    # --------------------------------------------------------

    print_top_counties(
        county,
        n=10,
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    section("OUTPUT CREATED")

    print(
        f"County risk:"
        f"\n{COUNTY_RISK_FILE}"
    )

    print(
        f"\nValidation report:"
        f"\n{COUNTY_VALIDATION_FILE}"
    )

    print(
        f"\nMember coverage:"
        f"\n{COUNTY_MEMBER_COVERAGE_FILE}"
    )

    print()

    resolved = int(
        member_risk[COUNTY_COLUMN]
        .notna()
        .sum()
    )

    unresolved = int(
        member_risk[COUNTY_COLUMN]
        .isna()
        .sum()
    )

    print(
        f"Members:              "
        f"{len(member_risk)}"
    )

    print(
        f"County resolved:      "
        f"{resolved}"
    )

    print(
        f"County unresolved:    "
        f"{unresolved}"
    )

    if unresolved > 0:

        print()
        print(
            "IMPORTANT:"
        )

        print(
            "Unresolved members were NOT assigned "
            "to any county."
        )

        print(
            "They remain available for member-level "
            "risk analysis."
        )

    section(
        "COUNTY RISK GENERATION COMPLETE"
    )


if __name__ == "__main__":
    main()