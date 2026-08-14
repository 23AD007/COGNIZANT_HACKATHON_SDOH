"""Create documented tract-level food-access features from normalized SRAM components."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_sram_features(components: pd.DataFrame) -> pd.DataFrame:
    required = {"tract_geoid", "state_fips", "county_fips", "tract_code", "State", "County20", "Urban", "POP2020", "PovertyRate", "LowIncomeTracts", "TractHUNV", "TractSNAP", "DD_SRAM_LILATracts_1And10", "DD_SRAM_HUNVFlag", "DD_SRAM_LAPOP1_10", "DD_SRAM_lapop1share", "DD_SRAM_lahunv1", "DD_SRAM_lasnap1", "SD_SRAM_LILATracts_1And10", "SD_SRAM_HUNVFlag", "SD_SRAM_LAPOP1_10", "SD_SRAM_lapop1share", "SD_SRAM_lahunv1", "SD_SRAM_lasnap1"}
    missing = required - set(components.columns)
    if missing:
        raise ValueError(f"SRAM components missing fields: {sorted(missing)}")
    if components["tract_geoid"].duplicated().any():
        raise ValueError("Duplicate tract GEOIDs prevent a one-row-per-tract table.")
    features = components.rename(columns={
        "State": "state_name", "County20": "county_name", "Urban": "urban_flag", "POP2020": "population_2020",
        "PovertyRate": "poverty_rate_pct", "LowIncomeTracts": "low_income_tract_flag",
        "TractHUNV": "households_without_vehicle_count", "TractSNAP": "snap_households_count",
        "DD_SRAM_LILATracts_1And10": "driving_low_income_low_access_flag", "DD_SRAM_HUNVFlag": "driving_low_vehicle_access_flag",
        "DD_SRAM_LAPOP1_10": "driving_low_access_population_beyond_1mi_10mi_count", "DD_SRAM_lapop1share": "driving_low_access_population_beyond_1mi_pct",
        "DD_SRAM_lahunv1": "driving_no_vehicle_households_beyond_1mi_count", "DD_SRAM_lasnap1": "driving_snap_households_beyond_1mi_count",
        "SD_SRAM_LILATracts_1And10": "straight_low_income_low_access_flag", "SD_SRAM_HUNVFlag": "straight_low_vehicle_access_flag",
        "SD_SRAM_LAPOP1_10": "straight_low_access_population_beyond_1mi_10mi_count", "SD_SRAM_lapop1share": "straight_low_access_population_beyond_1mi_pct",
        "SD_SRAM_lahunv1": "straight_no_vehicle_households_beyond_1mi_count", "SD_SRAM_lasnap1": "straight_snap_households_beyond_1mi_count",
    })
    columns = ["tract_geoid", "state_fips", "county_fips", "tract_code", "state_name", "county_name", "urban_flag", "population_2020", "poverty_rate_pct", "low_income_tract_flag", "households_without_vehicle_count", "snap_households_count", "driving_low_income_low_access_flag", "driving_low_vehicle_access_flag", "driving_low_access_population_beyond_1mi_10mi_count", "driving_low_access_population_beyond_1mi_pct", "driving_no_vehicle_households_beyond_1mi_count", "driving_snap_households_beyond_1mi_count", "straight_low_income_low_access_flag", "straight_low_vehicle_access_flag", "straight_low_access_population_beyond_1mi_10mi_count", "straight_low_access_population_beyond_1mi_pct", "straight_no_vehicle_households_beyond_1mi_count", "straight_snap_households_beyond_1mi_count"]
    features = features[columns].copy()
    if not features["tract_geoid"].str.fullmatch(r"\d{11}", na=False).all():
        raise ValueError("Final SRAM tract GEOIDs must be 11 digits.")
    nonnegative = [column for column in features if column.endswith("_count") or column.startswith("driving_") or column.startswith("straight_")]
    if (features[nonnegative] < 0).any().any():
        raise ValueError("SRAM distance/access values must be non-negative.")
    bounded = ["poverty_rate_pct", "driving_low_access_population_beyond_1mi_pct", "straight_low_access_population_beyond_1mi_pct"]
    if (~features[bounded].isna() & ~features[bounded].apply(lambda column: column.between(0, 100))).any().any():
        raise ValueError("SRAM percentage/rate values must be between 0 and 100.")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tract-level USDA SRAM food-access features.")
    parser.add_argument("--input-path", type=Path, default=Path("data/interim/sram_clean/sram_tract_components.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/sram_features.csv"))
    args = parser.parse_args()
    components = pd.read_csv(args.input_path, dtype={"tract_geoid": "string", "state_fips": "string", "county_fips": "string", "tract_code": "string"})
    features = build_sram_features(components)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_path, index=False)
    print(f"Saved {len(features)} SRAM tract feature rows to {args.output_path}")


if __name__ == "__main__":
    main()
