"""Convert normalized CDC PLACES prevalence records into one county row."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FEATURE_NAMES = {
    "ACCESS2": "places_uninsured_pct", "CASTHMA": "places_asthma_pct", "CHECKUP": "places_routine_checkup_pct",
    "CHD": "places_heart_disease_pct", "CHOLSCREEN": "places_cholesterol_screening_pct",
    "COLON_SCREEN": "places_colorectal_screening_pct", "COPD": "places_copd_pct",
    "CSMOKING": "places_smoking_pct", "DENTAL": "places_dental_visit_pct", "DIABETES": "places_diabetes_pct",
    "LPA": "places_physical_inactivity_pct", "MAMMOUSE": "places_mammography_pct",
    "MHLTH": "places_poor_mental_health_pct", "OBESITY": "places_obesity_pct",
    "PHLTH": "places_poor_physical_health_pct", "STROKE": "places_stroke_pct",
}
KEY_COLUMNS = ["county_fips", "county_name", "state_abbr", "state_name", "places_year", "prevalence_type"]


def build_places_features(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot valid PLACES measure values into a unique county feature table."""
    required = set(KEY_COLUMNS + ["measure_id", "prevalence_pct"])
    missing = required - set(long.columns)
    if missing:
        raise ValueError(f"Normalized PLACES data missing columns: {sorted(missing)}")
    if long.duplicated(["county_fips", "places_year", "prevalence_type", "measure_id"]).any():
        raise ValueError("Duplicate county/measure records prevent a one-row-per-county feature table.")
    invalid = long["prevalence_pct"].notna() & ~long["prevalence_pct"].between(0, 100)
    if invalid.any():
        raise ValueError("PLACES prevalence values must be between 0 and 100.")

    selected = long.loc[long["measure_id"].isin(FEATURE_NAMES)].copy()
    selected["feature_name"] = selected["measure_id"].map(FEATURE_NAMES)
    features = selected.pivot(index=KEY_COLUMNS, columns="feature_name", values="prevalence_pct").reset_index()
    features.columns.name = None
    if features["county_fips"].duplicated().any():
        raise ValueError("Duplicate county FIPS values found in final PLACES features.")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Create county-level CDC PLACES features.")
    parser.add_argument("--input-path", type=Path, default=Path("data/interim/places_clean/places_long.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/places_features.csv"))
    args = parser.parse_args()
    long = pd.read_csv(args.input_path, dtype={"county_fips": "string"})
    features = build_places_features(long)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_path, index=False)
    print(f"Saved {len(features)} county PLACES feature rows to {args.output_path}")


if __name__ == "__main__":
    main()
