"""
REPAIR MEMBER SDOH GEOGRAPHY

Purpose
-------
Repair / rebuild member-level SDOH geography by:

1. Loading Synthea patients.
2. Loading member SDOH features.
3. Loading county-level SDOH features.
4. Recovering county FIPS when it is explicitly available.
5. Never assigning an arbitrary county.
6. Leaving unresolved members as county_fips = NaN.
7. Rebuilding member-level SDOH model features.
8. Producing validation and repair reports.

Important
---------
A missing county is NOT replaced with another county merely because
that county is common in the dataset.

This is intentional because county-level SDOH variables are geographic
features and arbitrary geographic assignment would introduce false data.

Run from project root:

    py -3.12 -m src.modeling.repair_model_sdoh
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

RAW_DIR = ROOT / "data" / "raw"
SYNTHEA_DIR = RAW_DIR / "synthea"

PROCESSED_DIR = ROOT / "data" / "processed"


PATIENTS_FILE = SYNTHEA_DIR / "patients.csv"

MEMBER_SDOH_FILE = (
    PROCESSED_DIR / "member_sdoh_features.csv"
)

COUNTY_FEATURES_FILE = (
    PROCESSED_DIR / "county_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "member_sdoh_model_features.csv"
)

REPAIR_REPORT_FILE = (
    PROCESSED_DIR / "member_sdoh_geography_repair_report.csv"
)

COVERAGE_REPORT_FILE = (
    PROCESSED_DIR / "member_sdoh_geography_coverage.csv"
)

UNRESOLVED_FILE = (
    PROCESSED_DIR / "members_missing_county.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

MEMBER_ID_CANDIDATES = [
    "member_id",
    "patient_id",
    "Id",
    "ID",
    "PATIENT",
    "patient",
]

PATIENT_ID_CANDIDATES = [
    "patient_id",
    "member_id",
    "Id",
    "ID",
]

STATE_CANDIDATES = [
    "STATE",
    "state",
    "state_abbr",
    "STATE_ABBR",
    "STATE_NORM",
]

COUNTY_CANDIDATES = [
    "COUNTY",
    "county",
    "county_name",
    "COUNTY_NAME",
    "COUNTY_NORM",
]

ZIP_CANDIDATES = [
    "ZIP",
    "zip",
    "ZIPCODE",
    "zipcode",
    "zip_code",
    "ZIP_NORM",
]

FIPS_CANDIDATES = [
    "county_fips",
    "COUNTY_FIPS",
    "fips",
    "FIPS",
    "county_geoid",
    "COUNTY_GEOID",
]

# Columns that should never be used as model predictors.
NON_PREDICTOR_COLUMNS = {
    "member_id",
    "patient_id",
    "county_fips",
    "zip",
    "fips",
    "lat",
    "lon",
    "index_date",
}

TARGET_COLUMNS = {
    "target_inpatient_any",
    "target_emergency_any",
    "target_acute_any",
    "target_acute_2plus",
    "target_top25_utilization",
    "selected_target",
    "target_definition",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)

    return pd.read_csv(
        path,
        low_memory=False,
    )


def first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:

    for column in candidates:
        if column in df.columns:
            return column

    return None


def normalize_text(value) -> str | pd.NA:

    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if not value:
        return pd.NA

    return value.upper()


def normalize_member_id(
    series: pd.Series,
) -> pd.Series:

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
    )


def normalize_zip(
    series: pd.Series,
) -> pd.Series:

    result = (
        series.astype("string")
        .str.strip()
        .str.upper()
    )

    # Remove ZIP+4 suffix.
    result = result.str.split("-").str[0]

    # Keep first five numeric digits.
    result = result.str.extract(
        r"(\d{5})",
        expand=False,
    )

    return result


def normalize_state(
    series: pd.Series,
) -> pd.Series:

    return (
        series.astype("string")
        .str.strip()
        .str.upper()
    )


def normalize_county_name(
    series: pd.Series,
) -> pd.Series:

    result = (
        series.astype("string")
        .str.strip()
        .str.upper()
    )

    result = (
        result
        .str.replace(
            r"\bCOUNTY\b",
            "",
            regex=True,
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    return result


def normalize_fips(
    series: pd.Series,
) -> pd.Series:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric.round().astype("Int64")


# ============================================================
# COLUMN STANDARDIZATION
# ============================================================

def standardize_patient_columns(
    patients: pd.DataFrame,
) -> pd.DataFrame:

    df = patients.copy()

    member_col = first_existing_column(
        df,
        MEMBER_ID_CANDIDATES,
    )

    if member_col is None:
        raise ValueError(
            "Could not identify patient/member ID column "
            f"in patients.csv.\nAvailable columns:\n"
            f"{df.columns.tolist()}"
        )

    df["member_id"] = normalize_member_id(
        df[member_col]
    )

    state_col = first_existing_column(
        df,
        STATE_CANDIDATES,
    )

    county_col = first_existing_column(
        df,
        COUNTY_CANDIDATES,
    )

    zip_col = first_existing_column(
        df,
        ZIP_CANDIDATES,
    )

    fips_col = first_existing_column(
        df,
        FIPS_CANDIDATES,
    )

    if state_col:
        df["STATE_NORM"] = normalize_state(
            df[state_col]
        )
    else:
        df["STATE_NORM"] = pd.NA

    if county_col:
        df["COUNTY_NORM"] = normalize_county_name(
            df[county_col]
        )
    else:
        df["COUNTY_NORM"] = pd.NA

    if zip_col:
        df["ZIP_NORM"] = normalize_zip(
            df[zip_col]
        )
    else:
        df["ZIP_NORM"] = pd.NA

    if fips_col:
        df["county_fips"] = normalize_fips(
            df[fips_col]
        )
    else:
        df["county_fips"] = pd.Series(
            pd.array(
                [pd.NA] * len(df),
                dtype="Int64",
            ),
            index=df.index,
        )

    return df


def standardize_member_sdoh(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    member_col = first_existing_column(
        result,
        MEMBER_ID_CANDIDATES,
    )

    if member_col is None:
        raise ValueError(
            "member_sdoh_features.csv does not contain a "
            "recognizable member identifier."
        )

    if member_col != "member_id":
        result["member_id"] = normalize_member_id(
            result[member_col]
        )
    else:
        result["member_id"] = normalize_member_id(
            result["member_id"]
        )

    if "county_fips" in result.columns:
        result["county_fips"] = normalize_fips(
            result["county_fips"]
        )
    else:
        result["county_fips"] = pd.Series(
            pd.array(
                [pd.NA] * len(result),
                dtype="Int64",
            ),
            index=result.index,
        )

    return result


def standardize_county_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    fips_col = first_existing_column(
        result,
        FIPS_CANDIDATES,
    )

    if fips_col is None:
        raise ValueError(
            "county_features.csv does not contain county FIPS."
        )

    if fips_col != "county_fips":
        result["county_fips"] = normalize_fips(
            result[fips_col]
        )
    else:
        result["county_fips"] = normalize_fips(
            result["county_fips"]
        )

    result = result[
        result["county_fips"].notna()
    ].copy()

    result = result.drop_duplicates(
        subset=["county_fips"],
        keep="first",
    )

    return result


# ============================================================
# LOAD DATA
# ============================================================

def load_inputs():

    print_header("REPAIRING MEMBER SDOH GEOGRAPHY")

    patients = read_csv(
        PATIENTS_FILE
    )

    member_sdoh = read_csv(
        MEMBER_SDOH_FILE
    )

    county_features = read_csv(
        COUNTY_FEATURES_FILE
    )

    print(f"Patients:        {len(patients)}")
    print(f"Member SDOH:     {len(member_sdoh)}")
    print(f"County features: {len(county_features)}")

    patients = standardize_patient_columns(
        patients
    )

    member_sdoh = standardize_member_sdoh(
        member_sdoh
    )

    county_features = standardize_county_features(
        county_features
    )

    return (
        patients,
        member_sdoh,
        county_features,
    )


# ============================================================
# PATIENT → COUNTY MAPPING
# ============================================================

def build_patient_geography(
    patients: pd.DataFrame,
    member_sdoh: pd.DataFrame,
) -> pd.DataFrame:

    print_header(
        "BUILDING PATIENT → COUNTY MAPPING"
    )

    members = (
        member_sdoh[
            ["member_id"]
        ]
        .drop_duplicates()
        .copy()
    )

    geo_columns = [
        "member_id",
        "STATE_NORM",
        "COUNTY_NORM",
        "ZIP_NORM",
        "county_fips",
    ]

    patient_geo = patients[
        geo_columns
    ].copy()

    patient_geo = patient_geo.drop_duplicates(
        subset=["member_id"],
        keep="first",
    )

    mapping = members.merge(
        patient_geo,
        on="member_id",
        how="left",
    )

    print(
        f"Members:         {len(mapping)}"
    )

    print(
        "Missing county: "
        f"{mapping['county_fips'].isna().sum()}"
    )

    print(
        "Known county:    "
        f"{mapping['county_fips'].notna().sum()}"
    )

    return mapping


# ============================================================
# REPAIR MEMBER SDOH
# ============================================================

def repair_member_sdoh(
    member_sdoh: pd.DataFrame,
    county_features: pd.DataFrame,
    geography: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print_header(
        "REPAIRING MEMBER SDOH"
    )

    df = member_sdoh.copy()

    # --------------------------------------------------------
    # Remove old county-level feature columns.
    # --------------------------------------------------------

    existing_county_columns = [
        column
        for column in county_features.columns
        if column != "county_fips"
    ]

    for column in existing_county_columns:

        if column in df.columns:
            df = df.drop(
                columns=[column]
            )

    # --------------------------------------------------------
    # Attach best available county.
    # --------------------------------------------------------

    geo_map = geography[
        [
            "member_id",
            "county_fips",
        ]
    ].copy()

    geo_map = geo_map.drop_duplicates(
        subset=["member_id"]
    )

    df = df.drop(
        columns=["county_fips"],
        errors="ignore",
    )

    df = df.merge(
        geo_map,
        on="member_id",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Attach county SDOH.
    # --------------------------------------------------------

    county_merge = county_features[
        [
            "county_fips",
            *existing_county_columns,
        ]
    ].copy()

    df = df.merge(
        county_merge,
        on="county_fips",
        how="left",
        validate="many_to_one",
    )

    return df, geography


# ============================================================
# REPORT
# ============================================================

def create_repair_report(
    geography: pd.DataFrame,
    repaired_df: pd.DataFrame,
) -> pd.DataFrame:

    print_header(
        "CREATING GEOGRAPHY REPAIR REPORT"
    )

    # IMPORTANT:
    #
    # Do NOT access patients["member_id", ...] here.
    #
    # The standardized geography dataframe already contains
    # these columns.
    #
    # This fixes the KeyError from the previous implementation.

    report = geography[
        [
            "member_id",
            "STATE_NORM",
            "COUNTY_NORM",
            "ZIP_NORM",
            "county_fips",
        ]
    ].copy()

    report["county_status"] = np.where(
        report["county_fips"].notna(),
        "resolved",
        "unresolved",
    )

    # Count county-derived feature availability.

    feature_columns = [
        column
        for column in repaired_df.columns
        if column not in {
            "member_id",
            "county_fips",
        }
        and column not in TARGET_COLUMNS
        and column not in NON_PREDICTOR_COLUMNS
    ]

    if feature_columns:

        report["sdoh_feature_cells_missing"] = (
            repaired_df[
                feature_columns
            ]
            .isna()
            .sum(axis=1)
            .values
        )

        report["sdoh_features_missing_count"] = (
            repaired_df[
                feature_columns
            ]
            .isna()
            .sum(axis=1)
            .values
        )

    else:

        report["sdoh_feature_cells_missing"] = 0
        report["sdoh_features_missing_count"] = 0

    report["county_features_available"] = (
        report["county_fips"].notna()
    )

    return report


# ============================================================
# COVERAGE REPORT
# ============================================================

def create_coverage_report(
    geography: pd.DataFrame,
    repaired_df: pd.DataFrame,
) -> pd.DataFrame:

    total_members = len(geography)

    resolved = int(
        geography["county_fips"].notna().sum()
    )

    unresolved = total_members - resolved

    coverage_pct = (
        resolved / total_members * 100
        if total_members
        else 0
    )

    report = pd.DataFrame(
        [
            {
                "metric": "total_members",
                "value": total_members,
            },
            {
                "metric": "county_resolved_members",
                "value": resolved,
            },
            {
                "metric": "county_unresolved_members",
                "value": unresolved,
            },
            {
                "metric": "county_resolution_pct",
                "value": round(
                    coverage_pct,
                    2,
                ),
            },
            {
                "metric": "output_rows",
                "value": len(repaired_df),
            },
            {
                "metric": "output_columns",
                "value": len(repaired_df.columns),
            },
        ]
    )

    return report


# ============================================================
# VALIDATION
# ============================================================

def validate_output(
    repaired_df: pd.DataFrame,
    geography: pd.DataFrame,
) -> None:

    print_header(
        "VALIDATING REPAIRED MEMBER SDOH"
    )

    if repaired_df.empty:
        raise ValueError(
            "Repaired member SDOH dataset is empty."
        )

    if repaired_df["member_id"].duplicated().any():

        duplicates = (
            repaired_df.loc[
                repaired_df["member_id"].duplicated(
                    keep=False
                ),
                "member_id",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate member IDs after repair:\n"
            f"{duplicates}"
        )

    if repaired_df["member_id"].isna().any():
        raise ValueError(
            "Output contains missing member_id."
        )

    if len(repaired_df) != len(geography):
        raise ValueError(
            "Member count changed during repair.\n"
            f"Before: {len(geography)}\n"
            f"After:  {len(repaired_df)}"
        )

    missing_county = int(
        repaired_df["county_fips"].isna().sum()
    )

    resolved_county = int(
        repaired_df["county_fips"].notna().sum()
    )

    print(
        f"Members:             {len(repaired_df)}"
    )

    print(
        f"County resolved:     {resolved_county}"
    )

    print(
        f"County unresolved:   {missing_county}"
    )

    if missing_county:

        print()
        print(
            "WARNING:"
        )
        print(
            "Some members could not be assigned a "
            "defensible county."
        )
        print(
            "They remain county_fips = NaN."
        )

    infinite_count = 0

    numeric_columns = repaired_df.select_dtypes(
        include=[np.number]
    ).columns

    if len(numeric_columns):

        infinite_count = int(
            np.isinf(
                repaired_df[
                    numeric_columns
                ].to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
            ).sum()
        )

    print(
        f"Infinite numeric values: {infinite_count}"
    )

    if infinite_count:
        raise ValueError(
            "Infinite numeric values found in "
            "repaired dataset."
        )

    print(
        "Repair validation: PASSED"
    )


# ============================================================
# UNRESOLVED MEMBER FILE
# ============================================================

def create_unresolved_member_file(
    geography: pd.DataFrame,
) -> pd.DataFrame:

    unresolved = geography.loc[
        geography["county_fips"].isna(),
        [
            "member_id",
            "STATE_NORM",
            "COUNTY_NORM",
            "ZIP_NORM",
        ],
    ].copy()

    return unresolved


# ============================================================
# MAIN
# ============================================================

def main():

    (
        patients,
        member_sdoh,
        county_features,
    ) = load_inputs()

    # --------------------------------------------------------
    # Build patient → county mapping.
    # --------------------------------------------------------

    geography = build_patient_geography(
        patients=patients,
        member_sdoh=member_sdoh,
    )

    # --------------------------------------------------------
    # Repair member SDOH.
    # --------------------------------------------------------

    repaired_df, geography = repair_member_sdoh(
        member_sdoh=member_sdoh,
        county_features=county_features,
        geography=geography,
    )

    # --------------------------------------------------------
    # Validate.
    # --------------------------------------------------------

    validate_output(
        repaired_df=repaired_df,
        geography=geography,
    )

    # --------------------------------------------------------
    # Create reports.
    # --------------------------------------------------------

    repair_report = create_repair_report(
        geography=geography,
        repaired_df=repaired_df,
    )

    coverage_report = create_coverage_report(
        geography=geography,
        repaired_df=repaired_df,
    )

    unresolved_members = (
        create_unresolved_member_file(
            geography
        )
    )

    # --------------------------------------------------------
    # Save output.
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    repaired_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    repair_report.to_csv(
        REPAIR_REPORT_FILE,
        index=False,
    )

    coverage_report.to_csv(
        COVERAGE_REPORT_FILE,
        index=False,
    )

    unresolved_members.to_csv(
        UNRESOLVED_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print_header(
        "REPAIR COMPLETE"
    )

    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    print(
        f"Repair report:\n{REPAIR_REPORT_FILE}"
    )

    print(
        f"Coverage report:\n{COVERAGE_REPORT_FILE}"
    )

    print(
        f"Unresolved members:\n{UNRESOLVED_FILE}"
    )

    print()

    print(
        f"Members:              {len(repaired_df)}"
    )

    print(
        "County resolved:      "
        f"{repaired_df['county_fips'].notna().sum()}"
    )

    print(
        "County unresolved:    "
        f"{repaired_df['county_fips'].isna().sum()}"
    )

    print()

    if len(unresolved_members):

        print(
            "The following members remain without "
            "defensible county assignment:"
        )

        for member_id in unresolved_members[
            "member_id"
        ].tolist():

            print(
                f"  {member_id}"
            )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Do not manually assign these members to "
        "25017 or another county."
    )
    print(
        "Their county-dependent SDOH values remain "
        "missing and should be handled by the "
        "modeling/imputation pipeline."
    )


if __name__ == "__main__":
    main()