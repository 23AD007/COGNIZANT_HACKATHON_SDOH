from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .canonical_schema import (
    PROJECT_ROOT,
    load_existing_member_schema,
)
from .source_adapter import DataFrameSourceAdapter
from .validation import validate_source_dataframe


MODEL_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member_model_features.csv"
)

EXISTING_MEMBER_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member_model_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "canonical"
)

OUTPUT_DATASET = (
    OUTPUT_DIR
    / "canonical_member_features.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "canonical_member_validation.json"
)


def load_existing_member_dataset() -> pd.DataFrame:
    if not EXISTING_MEMBER_DATASET.exists():
        raise FileNotFoundError(
            "Existing member model dataset not found:\n"
            f"{EXISTING_MEMBER_DATASET}"
        )

    df = pd.read_csv(EXISTING_MEMBER_DATASET)

    if df.empty:
        raise ValueError(
            "Existing member model dataset is empty:\n"
            f"{EXISTING_MEMBER_DATASET}"
        )

    return df


def build_canonical_dataset() -> tuple[pd.DataFrame, dict]:
    print("=" * 78)
    print("HEALTHLENS CANONICAL MEMBER DATASET")
    print("=" * 78)

    # ---------------------------------------------------------------
    # 1. Load the ACTUAL existing HealthLens feature contract.
    # ---------------------------------------------------------------
    schema = load_existing_member_schema(
        MODEL_FEATURES_FILE
    )

    print(
        f"Schema source:       {MODEL_FEATURES_FILE}"
    )
    print(
        f"Member ID column:    {schema.member_id_column}"
    )
    print(
        f"Model features:      {len(schema.model_features)}"
    )

    # ---------------------------------------------------------------
    # 2. Load the ACTUAL existing member dataset.
    # ---------------------------------------------------------------
    source_df = load_existing_member_dataset()

    print(
        f"Existing members:    {len(source_df)}"
    )
    print(
        f"Existing columns:    {len(source_df.columns)}"
    )

    # ---------------------------------------------------------------
    # 3. Validate before adaptation.
    # ---------------------------------------------------------------
    source_validation = validate_source_dataframe(
        source_df,
        schema,
    )

    print(
        "Source validation:   "
        f"{'PASS' if source_validation.passed else 'FAIL'}"
    )

    if source_validation.errors:
        for error in source_validation.errors:
            print(f"  ERROR: {error}")

        raise ValueError(
            "Existing member dataset failed canonical "
            "source validation."
        )

    if source_validation.warnings:
        for warning in source_validation.warnings:
            print(f"  WARNING: {warning}")

    # ---------------------------------------------------------------
    # 4. Adapt using ONLY existing canonical model columns.
    #
    # No guessed mappings.
    # No synthetic values.
    # No filling missing attributes.
    # ---------------------------------------------------------------
    adapter = DataFrameSourceAdapter(
        schema=schema,
        source_name="existing_member_model_dataset",
    )

    records = adapter.adapt(source_df)

    print(
        f"Canonical records:   {len(records)}"
    )

    # ---------------------------------------------------------------
    # 5. Construct canonical dataframe.
    # ---------------------------------------------------------------
    rows: list[dict] = []

    for record in records:
        canonical_features = record.provenance.get(
            "canonical_features",
            {},
        )

        row = {
            schema.member_id_column: record.member_id
        }

        for feature in schema.model_features:
            row[feature] = canonical_features.get(
                feature,
                pd.NA,
            )

        rows.append(row)

    canonical_df = pd.DataFrame(rows)

    expected_columns = [
        schema.member_id_column,
        *schema.model_features,
    ]

    canonical_df = canonical_df.reindex(
        columns=expected_columns
    )

    # ---------------------------------------------------------------
    # 6. Regression checks.
    # ---------------------------------------------------------------
    regression = compare_with_existing_dataset(
        source_df=source_df,
        canonical_df=canonical_df,
        schema=schema,
    )

    # ---------------------------------------------------------------
    # 7. Serialize.
    # ---------------------------------------------------------------
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical_df.to_csv(
        OUTPUT_DATASET,
        index=False,
    )

    report = {
        "schema_source": str(MODEL_FEATURES_FILE),
        "source_dataset": str(EXISTING_MEMBER_DATASET),
        "output_dataset": str(OUTPUT_DATASET),
        "member_count": len(canonical_df),
        "feature_count": len(schema.model_features),
        "source_validation": source_validation.to_dict(),
        "regression": regression,
    }

    with OUTPUT_REPORT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    # ---------------------------------------------------------------
    # 8. Final result.
    # ---------------------------------------------------------------
    if not regression["passed"]:
        raise AssertionError(
            "Canonical dataset regression validation failed."
        )

    print()
    print("Canonical dataset:   PASS")
    print("Regression check:    PASS")
    print("Serialization:       PASS")

    print()
    print("Output:")
    print(f"  {OUTPUT_DATASET}")
    print(f"  {OUTPUT_REPORT}")

    print()
    print("=" * 78)
    print("CANONICAL MEMBER DATASET SELF-TEST: PASSED")
    print("=" * 78)

    return canonical_df, report


def compare_with_existing_dataset(
    source_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    schema,
) -> dict:

    result = {
        "passed": True,
        "source_rows": len(source_df),
        "canonical_rows": len(canonical_df),
        "source_features": len(schema.model_features),
        "canonical_features": len(schema.model_features),
        "member_id_match": False,
        "feature_columns_match": False,
        "value_mismatches": 0,
        "missing_canonical_values": 0,
        "errors": [],
    }

    member_column = schema.member_id_column

    # ---------------------------------------------------------------
    # Row count
    # ---------------------------------------------------------------
    if len(source_df) != len(canonical_df):
        result["errors"].append(
            "Canonical member count differs from source."
        )

    # ---------------------------------------------------------------
    # Member IDs
    # ---------------------------------------------------------------
    source_ids = (
        source_df[member_column]
        .astype(str)
        .tolist()
    )

    canonical_ids = (
        canonical_df[member_column]
        .astype(str)
        .tolist()
    )

    result["member_id_match"] = (
        source_ids == canonical_ids
    )

    if not result["member_id_match"]:
        result["errors"].append(
            "Canonical member ID ordering/content differs "
            "from source dataset."
        )

    # ---------------------------------------------------------------
    # Feature columns
    # ---------------------------------------------------------------
    expected_columns = [
        member_column,
        *schema.model_features,
    ]

    result["feature_columns_match"] = (
        list(canonical_df.columns)
        == expected_columns
    )

    if not result["feature_columns_match"]:
        result["errors"].append(
            "Canonical feature columns do not match "
            "the existing model feature schema."
        )

    # ---------------------------------------------------------------
    # Compare values.
    #
    # Only columns actually present in the source are compared.
    # Missing values remain missing.
    # ---------------------------------------------------------------
    common_features = [
        feature
        for feature in schema.model_features
        if feature in source_df.columns
    ]

    mismatches = 0
    missing_values = 0

    for feature in common_features:

        source_values = source_df[feature]
        canonical_values = canonical_df[feature]

        for source_value, canonical_value in zip(
            source_values,
            canonical_values,
        ):

            source_missing = pd.isna(source_value)
            canonical_missing = pd.isna(canonical_value)

            if source_missing:
                if not canonical_missing:
                    mismatches += 1
                continue

            if canonical_missing:
                missing_values += 1
                continue

            if not values_equal(
                source_value,
                canonical_value,
            ):
                mismatches += 1

    result["value_mismatches"] = mismatches
    result["missing_canonical_values"] = missing_values

    if mismatches:
        result["errors"].append(
            f"{mismatches} feature values changed "
            "during canonicalization."
        )

    result["passed"] = not result["errors"]

    return result


def values_equal(
    left,
    right,
) -> bool:

    # Numeric comparison
    try:
        left_float = float(left)
        right_float = float(right)

        if pd.notna(left_float) and pd.notna(right_float):
            return abs(left_float - right_float) <= 1e-12

    except (TypeError, ValueError):
        pass

    return str(left) == str(right)


if __name__ == "__main__":
    build_canonical_dataset()