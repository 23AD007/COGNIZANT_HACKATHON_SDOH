from __future__ import annotations

import pandas as pd

from .modeling_config import (
    FEATURE_FILE,
    TARGET_FILE,
    TARGET_COLUMN,
    ID_COLUMNS,
    GEOGRAPHY_COLUMNS,
    TARGET_LEAKAGE_COLUMNS,
)


def load_data():
    features = pd.read_csv(FEATURE_FILE)
    target = pd.read_csv(TARGET_FILE)

    if "member_id" not in features.columns:
        raise ValueError(
            "member_sdoh_model_features.csv must contain member_id"
        )

    if "patient_id" not in target.columns:
        raise ValueError(
            "member_clinical_target.csv must contain patient_id"
        )

    if TARGET_COLUMN not in target.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    target_small = target[
        ["patient_id", TARGET_COLUMN]
    ].copy()

    target_small[TARGET_COLUMN] = pd.to_numeric(
        target_small[TARGET_COLUMN],
        errors="raise",
    )

    if not set(
        target_small[TARGET_COLUMN].dropna().unique()
    ).issubset({0, 1}):
        raise ValueError(
            f"{TARGET_COLUMN} must contain only 0 and 1"
        )

    data = features.merge(
        target_small,
        left_on="member_id",
        right_on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise ValueError(
            "Feature/target merge returned zero rows."
        )

    if data["member_id"].duplicated().any():
        raise ValueError(
            "Duplicate member_id values after merge."
        )

    return data


def build_model_dataset():

    data = load_data()

    forbidden = (
        ID_COLUMNS
        | GEOGRAPHY_COLUMNS
        | TARGET_LEAKAGE_COLUMNS
    )

    excluded = sorted(
        set(data.columns) & forbidden
    )

    feature_columns = [
        column
        for column in data.columns
        if column not in forbidden
        and column != TARGET_COLUMN
    ]

    X = data[feature_columns].copy()

    # Keep only numeric predictors.
    X = X.select_dtypes(
        include="number"
    )

    if X.empty:
        raise ValueError(
            "No numeric predictors remain."
        )

    y = data[TARGET_COLUMN].astype(int)

    member_ids = data["member_id"].copy()

    county_fips = (
        data["county_fips"].copy()
        if "county_fips" in data.columns
        else pd.Series(
            index=data.index,
            dtype="object",
        )
    )

    leakage_remaining = (
        set(X.columns)
        & (
            ID_COLUMNS
            | GEOGRAPHY_COLUMNS
            | TARGET_LEAKAGE_COLUMNS
        )
    )

    if leakage_remaining:
        raise ValueError(
            "Leakage columns remain in X: "
            f"{sorted(leakage_remaining)}"
        )

    print("=" * 70)
    print("MODEL DATASET")
    print("=" * 70)

    print(f"Members: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(
        f"Positive target: {y.sum()} "
        f"({y.mean() * 100:.2f}%)"
    )

    print("\nExcluded columns:")
    for column in excluded:
        print(f"  {column}")

    print("\nModel features:")
    for column in X.columns:
        print(f"  {column}")

    return (
        X,
        y,
        member_ids,
        county_fips,
        data,
    )