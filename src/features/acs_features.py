"""Build a small, documented ACS SDOH feature table from cleaned ACS components."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TABLES = ("DP02", "DP03", "DP04", "DP05", "S0101", "S1501", "S1701", "S1901")


KEY_COLUMNS = ["GEO_ID", "NAME", "geographic_level"]


def build_acs_features(interim_dir: str | Path) -> pd.DataFrame:
    interim_dir = Path(interim_dir)
    merged: pd.DataFrame | None = None
    for table in TABLES:
        path = interim_dir / f"{table.lower()}_clean.csv"
        if not path.exists():
            raise FileNotFoundError(f"Cleaned ACS table not found: {path}")
        frame = pd.read_csv(path, dtype={"GEO_ID": "string", "NAME": "string"})
        if frame["GEO_ID"].duplicated().any():
            raise ValueError(f"Duplicate geographic keys found in {path.name}")
        merged = frame if merged is None else merged.merge(frame, on=KEY_COLUMNS, how="outer", validate="one_to_one")

    assert merged is not None
    features = merged[KEY_COLUMNS].copy()
    # Economic
    for column in ("poverty_pct", "unemployment_pct", "median_household_income", "public_assistance_pct", "uninsured_pct"):
        features[column] = merged[column]
    # Education
    features["education_less_than_high_school_pct"] = merged["education_less_than_9th_pct"] + merged["education_9th_to_12th_no_diploma_pct"]
    for column in ("education_high_school_pct", "education_college_pct", "education_bachelors_or_higher_pct"):
        features[column] = merged[column]
    # Housing and transportation
    features["housing_renter_pct"] = merged["housing_renter_pct"]
    features["housing_cost_burden_30pct_or_more"] = merged["housing_rent_30_to_34_9_pct"] + merged["housing_rent_35_plus_pct"]
    features["housing_crowded_pct"] = merged["housing_crowded_1_01_to_1_50_pct"] + merged["housing_crowded_1_51_plus_pct"]
    features["housing_vacancy_pct"] = merged["housing_vacancy_pct"]
    features["housing_no_vehicle_pct"] = merged["housing_no_vehicle_pct"]
    features["transport_no_vehicle_pct"] = merged["housing_no_vehicle_pct"]
    features["mean_commute_minutes"] = merged["mean_commute_minutes"]
    # Digital access is derived from documented household-access percentages.
    features["digital_no_computer_pct"] = 100 - merged["digital_with_computer_pct"]
    features["digital_no_broadband_pct"] = 100 - merged["digital_with_broadband_pct"]
    # Demographic context only; no race or ethnicity field is included.
    for column in ("population_total", "median_age_years", "female_pct"):
        features[column] = merged[column]
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Create documented ACS SDOH features.")
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim/acs_clean"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/acs_features.csv"))
    args = parser.parse_args()
    features = build_acs_features(args.interim_dir)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_path, index=False)
    print(f"Saved {len(features)} ACS geographic rows to {args.output_path}")


if __name__ == "__main__":
    main()
