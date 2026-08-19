from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"

SDOH_FILE = PROCESSED / "member_sdoh_model_features.csv"

CLINICAL_FILE = PROCESSED / "member_clinical_features.csv"

TARGET_FILE = PROCESSED / "member_clinical_target.csv"

OUTPUT_FILE = PROCESSED / "member_model_features.csv"

# Compatibility file
DATASET_ALIAS_FILE = PROCESSED / "member_model_dataset.csv"

VALIDATION_FILE = (
    PROCESSED / "member_model_dataset_validation.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

MEMBER_ID = "member_id"

PATIENT_ID = "patient_id"

TARGET = "target_inpatient_any"


# ------------------------------------------------------------
# Metadata / identifiers
# ------------------------------------------------------------
#
# These columns are retained in the final dataset.
#
# They are NOT treated as model predictors by this dataset
# construction script.
# ------------------------------------------------------------

IDENTIFIER_COLUMNS = {
    MEMBER_ID,
    PATIENT_ID,
    "county_fips",
    "zip",
    "fips",
    "lat",
    "lon",
}


# ------------------------------------------------------------
# Target leakage columns
# ------------------------------------------------------------
#
# These columns must never enter the predictor matrix because
# they directly describe utilization/outcome information.
# ------------------------------------------------------------

TARGET_LEAKAGE_COLUMNS = {
    "encounter_count",
    "emergency_count",
    "inpatient_count",
    "urgent_care_count",
    "target_emergency_any",
    "target_inpatient_any",
    "target_acute_any",
    "target_acute_2plus",
    "target_top25_utilization",
    "selected_target",
    "target_definition",
}


# ------------------------------------------------------------
# Categorical demographic / geographic columns
# ------------------------------------------------------------
#
# These must remain categorical.
#
# We should NOT convert:
#   gender -> numeric
#   race -> numeric
#   ethnicity -> numeric
#   city -> numeric
#   etc.
# ------------------------------------------------------------

CATEGORICAL_COLUMNS = {
    "birthdate",
    "gender",
    "race",
    "ethnicity",
    "marital_status",
    "city",
    "state",
    "county",
    "county_name",
    "state_abbr",
    "state_name",
    "prevalence_type",
}


# ============================================================
# HELPERS
# ============================================================

def normalize_id(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize UUID/string identifiers.
    """

    return (
        series
        .astype("string")
        .str.strip()
    )


def normalize_fips(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize county FIPS values.

    Examples:
        25017.0 -> 25017
        "25017"  -> 25017
        NaN      -> <NA>
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    rounded = numeric.round()

    return (
        rounded
        .astype("Int64")
        .astype("string")
        .str.zfill(5)
    )


def get_predictor_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return columns eligible to be model predictors.

    Identifiers, target, leakage columns and index_date
    are excluded.

    The actual columns remain in the final dataset.
    """

    excluded = (
        IDENTIFIER_COLUMNS
        | TARGET_LEAKAGE_COLUMNS
        | {"index_date"}
    )

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


def get_categorical_predictors(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return categorical predictors that exist in the dataframe.
    """

    predictors = get_predictor_columns(df)

    return [
        column
        for column in predictors
        if column in CATEGORICAL_COLUMNS
    ]


def get_numeric_predictors(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return predictors that are not categorical.
    """

    predictors = get_predictor_columns(df)

    categorical = set(
        get_categorical_predictors(df)
    )

    return [
        column
        for column in predictors
        if column not in categorical
    ]


# ============================================================
# LOAD
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING MEMBER-LEVEL DATA")
    print("=" * 70)

    for path in [
        SDOH_FILE,
        CLINICAL_FILE,
        TARGET_FILE,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    sdoh = pd.read_csv(
        SDOH_FILE,
        low_memory=False,
    )

    clinical = pd.read_csv(
        CLINICAL_FILE,
        low_memory=False,
    )

    target = pd.read_csv(
        TARGET_FILE,
        low_memory=False,
    )

    print(
        f"SDOH rows:      {len(sdoh)}"
    )

    print(
        f"SDOH columns:   {len(sdoh.columns)}"
    )

    print(
        f"Clinical rows:  {len(clinical)}"
    )

    print(
        f"Clinical columns:{len(clinical.columns)}"
    )

    print(
        f"Target rows:    {len(target)}"
    )

    print(
        f"Target columns: {len(target.columns)}"
    )

    return sdoh, clinical, target


# ============================================================
# NORMALIZE
# ============================================================

def normalize_data(
    sdoh: pd.DataFrame,
    clinical: pd.DataFrame,
    target: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("NORMALIZING IDENTIFIERS")
    print("=" * 70)

    # --------------------------------------------------------
    # SDOH
    # --------------------------------------------------------

    if MEMBER_ID in sdoh.columns:

        sdoh[MEMBER_ID] = normalize_id(
            sdoh[MEMBER_ID]
        )

    if PATIENT_ID in sdoh.columns:

        sdoh[PATIENT_ID] = normalize_id(
            sdoh[PATIENT_ID]
        )

    if "county_fips" in sdoh.columns:

        sdoh["county_fips"] = normalize_fips(
            sdoh["county_fips"]
        )

    # --------------------------------------------------------
    # Clinical
    # --------------------------------------------------------

    if MEMBER_ID in clinical.columns:

        clinical[MEMBER_ID] = normalize_id(
            clinical[MEMBER_ID]
        )

    if PATIENT_ID in clinical.columns:

        clinical[PATIENT_ID] = normalize_id(
            clinical[PATIENT_ID]
        )

    if "county_fips" in clinical.columns:

        clinical["county_fips"] = normalize_fips(
            clinical["county_fips"]
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if PATIENT_ID not in target.columns:

        raise ValueError(
            "Target file must contain patient_id."
        )

    target[PATIENT_ID] = normalize_id(
        target[PATIENT_ID]
    )

    return sdoh, clinical, target


# ============================================================
# VALIDATION HELPERS
# ============================================================

def check_unique(
    df: pd.DataFrame,
    column: str,
    name: str,
):

    if column not in df.columns:

        raise ValueError(
            f"{name} missing {column}"
        )

    duplicates = int(
        df[column].duplicated().sum()
    )

    print(
        f"{name} duplicate {column}: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            f"{name} contains duplicate "
            f"{column}."
        )


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    sdoh: pd.DataFrame,
    clinical: pd.DataFrame,
    target: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("VALIDATING INPUT STRUCTURE")
    print("=" * 70)

    check_unique(
        sdoh,
        MEMBER_ID,
        "SDOH",
    )

    check_unique(
        clinical,
        MEMBER_ID,
        "Clinical",
    )

    check_unique(
        target,
        PATIENT_ID,
        "Target",
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if TARGET not in target.columns:

        raise ValueError(
            f"Target file missing {TARGET}"
        )

    # --------------------------------------------------------
    # SDOH geography
    # --------------------------------------------------------

    if "county_fips" not in sdoh.columns:

        raise ValueError(
            "SDOH file must contain county_fips."
        )

    missing_county = int(
        sdoh["county_fips"].isna().sum()
    )

    print(
        f"SDOH missing county_fips: "
        f"{missing_county}"
    )

    if missing_county > 0:

        print()
        print("WARNING:")

        print(
            f"{missing_county} members do not have "
            "a defensible county assignment."
        )

        print(
            "They will remain county_fips = NaN."
        )

        print(
            "No arbitrary county assignment will be performed."
        )

    print("\nInput structure: OK")


# ============================================================
# PATIENT → MEMBER MAPPING
# ============================================================

def build_mapping(
    clinical: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("BUILDING PATIENT → MEMBER MAPPING")
    print("=" * 70)

    clinical = clinical.copy()

    # --------------------------------------------------------
    # Current synthetic data:
    #
    # member_id and patient_id are available.
    #
    # If patient_id is absent, create it from member_id.
    # --------------------------------------------------------

    if PATIENT_ID not in clinical.columns:

        print(
            "Clinical dataset does not contain patient_id."
        )

        print(
            "Creating patient_id from member_id "
            "for the synthetic member dataset."
        )

        clinical[PATIENT_ID] = (
            clinical[MEMBER_ID]
        )

    mapping = clinical[
        [
            PATIENT_ID,
            MEMBER_ID,
        ]
    ].copy()

    mapping[PATIENT_ID] = normalize_id(
        mapping[PATIENT_ID]
    )

    mapping[MEMBER_ID] = normalize_id(
        mapping[MEMBER_ID]
    )

    check_unique(
        mapping,
        PATIENT_ID,
        "Mapping",
    )

    print(
        f"Patient IDs mapped: "
        f"{len(mapping)}"
    )

    return clinical, mapping


# ============================================================
# TARGET MAPPING
# ============================================================

def attach_target(
    target: pd.DataFrame,
    mapping: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("MAPPING TARGET TO MEMBER_ID")
    print("=" * 70)

    result = target.merge(
        mapping,
        on=PATIENT_ID,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched = int(
        (result["_merge"] == "both").sum()
    )

    unmatched = int(
        (result["_merge"] != "both").sum()
    )

    print(
        f"Target rows:     {len(result)}"
    )

    print(
        f"Matched targets: {matched}"
    )

    print(
        f"Unmatched targets:{unmatched}"
    )

    if unmatched > 0:

        ids = result.loc[
            result["_merge"] != "both",
            PATIENT_ID,
        ].tolist()

        raise ValueError(
            "Unmatched target patient IDs:\n"
            f"{ids}"
        )

    result = result.drop(
        columns=["_merge"]
    )

    return result


# ============================================================
# TARGET VALIDATION
# ============================================================

def validate_target(
    target: pd.DataFrame,
):

    values = pd.to_numeric(
        target[TARGET],
        errors="coerce",
    )

    if values.isna().any():

        raise ValueError(
            f"{TARGET} contains missing values."
        )

    values = values.astype(int)

    if not set(values.unique()).issubset(
        {0, 1}
    ):

        raise ValueError(
            f"{TARGET} must contain only 0/1."
        )

    target[TARGET] = values

    positive = int(
        (values == 1).sum()
    )

    negative = int(
        (values == 0).sum()
    )

    print("\n" + "=" * 70)
    print("VALIDATING TARGET")
    print("=" * 70)

    print(
        f"Members:        {len(values)}"
    )

    print(
        f"Positive:       {positive}"
    )

    print(
        f"Negative:       {negative}"
    )

    print(
        f"Positive rate:  "
        f"{positive / len(values):.2%}"
    )

    return target


# ============================================================
# MERGE
# ============================================================

def merge_all(
    sdoh: pd.DataFrame,
    clinical: pd.DataFrame,
    target: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("MERGING SDOH + CLINICAL + TARGET")
    print("=" * 70)

    # --------------------------------------------------------
    # Clinical predictor columns
    # --------------------------------------------------------

    clinical_excluded = (
        IDENTIFIER_COLUMNS
        | TARGET_LEAKAGE_COLUMNS
        | {"index_date"}
    )

    clinical_predictors = [
        c
        for c in clinical.columns
        if c not in clinical_excluded
    ]

    print(
        f"Clinical predictor columns: "
        f"{len(clinical_predictors)}"
    )

    clinical_subset = clinical[
        [MEMBER_ID] + clinical_predictors
    ].copy()

    # --------------------------------------------------------
    # SDOH + Clinical
    # --------------------------------------------------------

    merged = sdoh.merge(
        clinical_subset,
        on=MEMBER_ID,
        how="inner",
        validate="one_to_one",
    )

    print(
        f"After SDOH + clinical merge: "
        f"{len(merged)} members"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_subset = target[
        [
            MEMBER_ID,
            TARGET,
        ]
    ].copy()

    merged = merged.merge(
        target_subset,
        on=MEMBER_ID,
        how="inner",
        validate="one_to_one",
    )

    print(
        f"After target merge: "
        f"{len(merged)} members"
    )

    return merged


# ============================================================
# FILL MISSING VALUES
# ============================================================

def fill_missing_values(
    df: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("FILLING MISSING PREDICTOR VALUES")
    print("=" * 70)

    df = df.copy()

    predictors = get_predictor_columns(df)

    categorical_predictors = (
        get_categorical_predictors(df)
    )

    numeric_predictors = (
        get_numeric_predictors(df)
    )

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------
    #
    # Three members have missing age.
    #
    # Their birthdates are available in the source member
    # dataset, so calculate age directly from birthdate.
    #
    # Project reference date:
    # 2026-08-18
    # --------------------------------------------------------

    if "age" in df.columns:

        missing_age_before = int(
            df["age"].isna().sum()
        )

        if missing_age_before > 0:

            birthdates = pd.to_datetime(
                df["birthdate"],
                errors="coerce",
            )

            reference_date = pd.Timestamp(
                "2026-08-18"
            )

            # Calculate age correctly using vectorized
            # Pandas operations.
            birthday_not_reached = (
                (
                    birthdates.dt.month
                    > reference_date.month
                )
                |
                (
                    (
                        birthdates.dt.month
                        == reference_date.month
                    )
                    &
                    (
                        birthdates.dt.day
                        > reference_date.day
                    )
                )
            )

            calculated_age = (
                reference_date.year
                - birthdates.dt.year
                - birthday_not_reached.astype(int)
            )

            df["age"] = df["age"].fillna(
                calculated_age
            )

            missing_age_after = int(
                df["age"].isna().sum()
            )

            filled_age = (
                missing_age_before
                - missing_age_after
            )

            print(
                f"Age values filled from birthdate: "
                f"{filled_age}"
            )

            if missing_age_after > 0:

                print(
                    f"WARNING: {missing_age_after} "
                    "age values remain missing because "
                    "birthdate was unavailable."
                )

    # --------------------------------------------------------
    # CATEGORICAL VALUES
    # --------------------------------------------------------
    #
    # Do not invent a demographic category.
    #
    # Missing marital status, for example, becomes:
    #     Unknown
    #
    # The original values M/S/D/W are preserved.
    # --------------------------------------------------------

    for column in categorical_predictors:

        missing_before = int(
            df[column].isna().sum()
        )

        if missing_before == 0:
            continue

        df[column] = (
            df[column]
            .astype("string")
            .fillna("Unknown")
        )

        print(
            f"{column}: filled "
            f"{missing_before} missing values "
            f"with 'Unknown'"
        )

    # --------------------------------------------------------
    # NUMERIC VALUES
    # --------------------------------------------------------
    #
    # For genuine numeric predictors, use the median.
    #
    # This is a preprocessing step only.
    # It does not change the underlying source datasets.
    # --------------------------------------------------------

    for column in numeric_predictors:

        missing_before = int(
            df[column].isna().sum()
        )

        if missing_before == 0:
            continue

        numeric_values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        median_value = numeric_values.median()

        if pd.isna(median_value):

            raise ValueError(
                f"Cannot impute {column}: "
                "all values are missing."
            )

        df[column] = (
            numeric_values
            .fillna(median_value)
        )

        print(
            f"{column}: filled "
            f"{missing_before} missing values "
            f"with median = {median_value}"
        )

    # --------------------------------------------------------
    # FINAL MISSING CHECK
    # --------------------------------------------------------

    remaining = (
        df[predictors]
        .isna()
        .sum()
    )

    remaining = remaining[
        remaining > 0
    ].sort_values(
        ascending=False
    )

    print(
        f"\nRemaining missing predictor columns: "
        f"{len(remaining)}"
    )

    if len(remaining) > 0:

        print(
            remaining.to_string()
        )

        raise ValueError(
            "Missing predictor values remain "
            "after imputation."
        )

    print(
        "All predictor values are filled."
    )

    return df


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_features(
    df: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("VALIDATING MODEL FEATURES")
    print("=" * 70)

    predictor_columns = get_predictor_columns(df)

    categorical_predictors = (
        get_categorical_predictors(df)
    )

    numeric_predictors = (
        get_numeric_predictors(df)
    )

    print(
        f"Total predictors: "
        f"{len(predictor_columns)}"
    )

    print(
        f"Numeric predictors: "
        f"{len(numeric_predictors)}"
    )

    print(
        f"Categorical predictors: "
        f"{len(categorical_predictors)}"
    )

    # --------------------------------------------------------
    # Validate numeric predictors
    # --------------------------------------------------------

    numeric_conversion_failures = []

    for column in numeric_predictors:

        original = df[column]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        failures = (
            original.notna()
            & converted.isna()
        )

        if failures.any():

            numeric_conversion_failures.append(
                column
            )

        df[column] = converted

    if numeric_conversion_failures:

        raise ValueError(
            "Non-numeric values detected in "
            "numeric predictors:\n"
            f"{numeric_conversion_failures}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = (
        df[predictor_columns]
        .isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    print(
        f"Features with missing values: "
        f"{len(missing)}"
    )

    if len(missing) > 0:

        print(
            missing.to_string()
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    infinite = []

    for column in numeric_predictors:

        values = df[column].dropna()

        if len(values) == 0:
            continue

        if np.isinf(
            values.to_numpy()
        ).any():

            infinite.append(
                column
            )

    print(
        f"Infinite-value features: "
        f"{len(infinite)}"
    )

    if infinite:

        print(
            "Infinite columns:"
        )

        for column in infinite:

            print(
                f"  {column}"
            )

        raise ValueError(
            "Infinite values detected "
            "in model predictors."
        )

    # --------------------------------------------------------
    # Constant features
    # --------------------------------------------------------

    constant = [
        c
        for c in predictor_columns
        if df[c].nunique(
            dropna=True
        ) <= 1
    ]

    print(
        f"Constant features: "
        f"{len(constant)}"
    )

    if constant:

        print(
            "Constant columns:"
        )

        for column in constant:

            print(
                f"  {column}"
            )

    # --------------------------------------------------------
    # County validation
    # --------------------------------------------------------

    if "county_fips" in df.columns:

        county_missing = int(
            df["county_fips"].isna().sum()
        )

        county_known = (
            len(df) - county_missing
        )

        print(
            f"County resolved: "
            f"{county_known}"
        )

        print(
            f"County unresolved: "
            f"{county_missing}"
        )

        if county_missing > 0:

            print()
            print("WARNING:")

            print(
                f"{county_missing} members remain "
                "without defensible county assignment."
            )

    # --------------------------------------------------------
    # Member uniqueness
    # --------------------------------------------------------

    if df[MEMBER_ID].duplicated().any():

        duplicates = (
            df.loc[
                df[MEMBER_ID].duplicated(
                    keep=False
                ),
                MEMBER_ID,
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate member_id in final dataset:\n"
            f"{duplicates}"
        )

    return (
        predictor_columns,
        missing,
        constant,
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

def leakage_check(
    df: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("TARGET LEAKAGE CHECK")
    print("=" * 70)

    if TARGET not in df.columns:

        raise ValueError(
            "Target missing from final dataset."
        )

    predictor_columns = get_predictor_columns(
        df
    )

    found = [
        column
        for column in TARGET_LEAKAGE_COLUMNS
        if column in predictor_columns
    ]

    if found:

        raise ValueError(
            "Potential target leakage columns found:\n"
            f"{found}"
        )

    print(
        f"Target column: {TARGET}"
    )

    print(
        "Target is kept separately from predictor columns."
    )

    print(
        "Leakage check: PASS"
    )


# ============================================================
# VALIDATION REPORT
# ============================================================

def create_report(
    df: pd.DataFrame,
):

    predictor_columns = get_predictor_columns(
        df
    )

    categorical_predictors = (
        get_categorical_predictors(df)
    )

    rows = []

    for column in predictor_columns:

        series = df[column]

        if column in categorical_predictors:

            rows.append(
                {
                    "feature": column,
                    "type": "categorical",
                    "rows": len(series),
                    "missing": int(
                        series.isna().sum()
                    ),
                    "missing_pct": float(
                        series.isna().mean() * 100
                    ),
                    "unique_values": int(
                        series.nunique()
                    ),
                    "mean": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )

        else:

            values = pd.to_numeric(
                series,
                errors="coerce",
            )

            non_missing = (
                values.dropna()
            )

            rows.append(
                {
                    "feature": column,
                    "type": "numeric",
                    "rows": len(series),
                    "missing": int(
                        values.isna().sum()
                    ),
                    "missing_pct": float(
                        values.isna().mean() * 100
                    ),
                    "unique_values": int(
                        values.nunique()
                    ),
                    "mean": float(
                        non_missing.mean()
                    )
                    if len(non_missing) > 0
                    else np.nan,
                    "min": float(
                        non_missing.min()
                    )
                    if len(non_missing) > 0
                    else np.nan,
                    "max": float(
                        non_missing.max()
                    )
                    if len(non_missing) > 0
                    else np.nan,
                }
            )

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    return report


# ============================================================
# FINAL DATASET VALIDATION
# ============================================================

def validate_final_dataset(
    df: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("FINAL DATASET VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    if len(df) != 1279:

        raise ValueError(
            "Expected 1279 members but obtained "
            f"{len(df)}."
        )

    # --------------------------------------------------------
    # Member IDs
    # --------------------------------------------------------

    if df[MEMBER_ID].isna().any():

        raise ValueError(
            "Final dataset contains missing member_id."
        )

    if df[MEMBER_ID].duplicated().any():

        raise ValueError(
            "Final dataset contains duplicate member_id."
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if df[TARGET].isna().any():

        raise ValueError(
            "Final dataset contains missing target."
        )

    target_values = set(
        df[TARGET].astype(int).unique()
    )

    if not target_values.issubset(
        {0, 1}
    ):

        raise ValueError(
            f"Invalid target values: "
            f"{target_values}"
        )

    # --------------------------------------------------------
    # County
    # --------------------------------------------------------

    county_missing = int(
        df["county_fips"].isna().sum()
    )

    print(
        f"Members:             {len(df)}"
    )

    print(
        f"County resolved:     "
        f"{len(df) - county_missing}"
    )

    print(
        f"County unresolved:   "
        f"{county_missing}"
    )

    print(
        f"Target missing:      "
        f"{int(df[TARGET].isna().sum())}"
    )

    # --------------------------------------------------------
    # Predictor groups
    # --------------------------------------------------------

    predictors = get_predictor_columns(df)

    categorical_predictors = (
        get_categorical_predictors(df)
    )

    numeric_predictors = (
        get_numeric_predictors(df)
    )

    # --------------------------------------------------------
    # Check categorical predictors
    # --------------------------------------------------------

    categorical_missing = (
        df[categorical_predictors]
        .isna()
        .sum()
    )

    categorical_missing = (
        categorical_missing[
            categorical_missing > 0
        ]
    )

    if len(categorical_missing) > 0:

        raise ValueError(
            "Missing categorical predictor values:\n"
            f"{categorical_missing.to_dict()}"
        )

    # --------------------------------------------------------
    # Check numeric predictors
    # --------------------------------------------------------

    numeric_conversion_failures = []

    for column in numeric_predictors:

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        failures = (
            df[column].notna()
            & converted.isna()
        )

        if failures.any():

            numeric_conversion_failures.append(
                column
            )

    if numeric_conversion_failures:

        raise ValueError(
            "Non-numeric values detected in "
            "numeric predictors:\n"
            f"{numeric_conversion_failures}"
        )

    # --------------------------------------------------------
    # Check ALL missing predictor cells
    # --------------------------------------------------------

    missing_predictors = (
        df[predictors]
        .isna()
        .sum()
    )

    total_missing_predictor_cells = int(
        missing_predictors.sum()
    )

    if total_missing_predictor_cells > 0:

        raise ValueError(
            "Missing predictor values remain:\n"
            f"{missing_predictors[missing_predictors > 0]}"
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    infinite_columns = []

    for column in numeric_predictors:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(values) == 0:
            continue

        if np.isinf(
            values.to_numpy()
        ).any():

            infinite_columns.append(
                column
            )

    if infinite_columns:

        raise ValueError(
            "Infinite predictor values detected:\n"
            f"{infinite_columns}"
        )

    print(
        "Missing predictor cells: 0"
    )

    print(
        "Final dataset validation: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("MEMBER MODEL DATASET CONSTRUCTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    sdoh, clinical, target = load_data()

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    sdoh, clinical, target = normalize_data(
        sdoh,
        clinical,
        target,
    )

    # --------------------------------------------------------
    # Validate input structure
    # --------------------------------------------------------

    validate_inputs(
        sdoh,
        clinical,
        target,
    )

    # --------------------------------------------------------
    # Patient → member mapping
    # --------------------------------------------------------

    clinical, mapping = build_mapping(
        clinical
    )

    # --------------------------------------------------------
    # Map target
    # --------------------------------------------------------

    target = attach_target(
        target,
        mapping,
    )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    target = validate_target(
        target
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = merge_all(
        sdoh,
        clinical,
        target,
    )

    # --------------------------------------------------------
    # Fill missing predictor values
    # --------------------------------------------------------

    merged = fill_missing_values(
        merged
    )

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    (
        predictor_columns,
        missing,
        constant,
    ) = validate_features(
        merged
    )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    leakage_check(
        merged
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    validate_final_dataset(
        merged
    )

    # --------------------------------------------------------
    # Validation report
    # --------------------------------------------------------

    create_report(
        merged
    )

    # --------------------------------------------------------
    # Save primary output
    # --------------------------------------------------------

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save compatibility alias
    # --------------------------------------------------------

    merged.to_csv(
        DATASET_ALIAS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    positive = int(
        merged[TARGET].sum()
    )

    negative = int(
        (merged[TARGET] == 0).sum()
    )

    county_missing = int(
        merged["county_fips"].isna().sum()
    )

    missing_predictor_cells = int(
        merged[predictor_columns]
        .isna()
        .sum()
        .sum()
    )

    print("\n" + "=" * 70)
    print("FINAL MEMBER MODEL DATASET")
    print("=" * 70)

    print(
        f"Members:          {len(merged)}"
    )

    print(
        f"Predictors:       {len(predictor_columns)}"
    )

    print(
        f"Positive target:  {positive}"
    )

    print(
        f"Negative target:  {negative}"
    )

    print(
        f"Positive rate:    "
        f"{positive / len(merged):.2%}"
    )

    print(
        f"Missing counties: "
        f"{county_missing}"
    )

    print(
        f"Missing predictor cells: "
        f"{missing_predictor_cells}"
    )

    print(
        f"Constant predictors: "
        f"{len(constant)}"
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"\nModel dataset:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nCompatibility alias:\n"
        f"{DATASET_ALIAS_FILE}"
    )

    print(
        f"\nValidation report:\n"
        f"{VALIDATION_FILE}"
    )

    print("\n" + "=" * 70)
    print("DATASET CONSTRUCTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()