"""Identifier-only assembly of county and tract community feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from src.geography.geography_mapping import acs_geography_keys, derive_county_keys, derive_county_keys_from_tract
except ModuleNotFoundError:  # Enables direct execution with `python src/geography/geographic_merge.py`.
    from geography_mapping import acs_geography_keys, derive_county_keys, derive_county_keys_from_tract


def assert_unique(frame: pd.DataFrame, key: str, source: str) -> int:
    duplicates = int(frame[key].duplicated(keep=False).sum()) if key in frame else 0
    if duplicates:
        raise ValueError(f"{source} has {duplicates} duplicate {key} records.")
    return duplicates


def canonicalize_places(places: pd.DataFrame) -> pd.DataFrame:
    if "county_fips" not in places:
        raise ValueError("PLACES features lack county_fips")
    keys = derive_county_keys(places["county_fips"])
    result = places.copy()
    for column in keys:
        result[column] = keys[column]
    assert_unique(result, "county_fips", "PLACES")
    return result


def canonicalize_acs(acs: pd.DataFrame) -> pd.DataFrame:
    if "GEO_ID" not in acs:
        raise ValueError("ACS features lack GEO_ID")
    result = acs.copy()
    keys = acs_geography_keys(result["GEO_ID"])
    for column in keys:
        result[column] = keys[column]
    result["tract_geoid"] = pd.Series(pd.NA, index=result.index, dtype="string")
    # GEO_ID, not a name field, remains the canonical ACS uniqueness criterion.
    assert_unique(result, "GEO_ID", "ACS")
    return result


def canonicalize_sram(sram: pd.DataFrame) -> pd.DataFrame:
    if "tract_geoid" not in sram:
        raise ValueError("SRAM features lack tract_geoid")
    result = sram.copy()
    keys = derive_county_keys_from_tract(result["tract_geoid"])
    for column in keys:
        result[column] = keys[column]
    assert_unique(result, "tract_geoid", "SRAM")
    return result


def build_join_report(acs: pd.DataFrame, places: pd.DataFrame, sram: pd.DataFrame, county_features: pd.DataFrame, tract_features: pd.DataFrame) -> pd.DataFrame:
    acs_county_present = acs["county_fips"].notna()
    places_matched = places["county_fips"].isin(acs.loc[acs_county_present, "county_fips"])
    rows = [
        {"dataset": "acs", "key": "county_fips", "total_geographic_records": len(acs), "matched_records": int(acs_county_present.sum()), "unmatched_records": int((~acs_county_present).sum()), "duplicate_keys": int(acs["GEO_ID"].duplicated().sum()), "missing_keys": int((~acs_county_present).sum())},
        {"dataset": "places", "key": "county_fips", "total_geographic_records": len(places), "matched_records": int(places_matched.sum()), "unmatched_records": int((~places_matched).sum()), "duplicate_keys": int(places["county_fips"].duplicated().sum()), "missing_keys": int(places["county_fips"].isna().sum())},
        {"dataset": "county_features", "key": "county_fips", "total_geographic_records": len(county_features), "matched_records": int((county_features["_merge"] == "both").sum()), "unmatched_records": int((county_features["_merge"] != "both").sum()), "duplicate_keys": int(county_features.loc[county_features["county_fips"].notna(), "county_fips"].duplicated().sum()), "missing_keys": int(county_features["county_fips"].isna().sum())},
        {"dataset": "sram", "key": "tract_geoid", "total_geographic_records": len(sram), "matched_records": len(sram), "unmatched_records": 0, "duplicate_keys": int(sram["tract_geoid"].duplicated().sum()), "missing_keys": int(sram["tract_geoid"].isna().sum())},
        {"dataset": "tract_features", "key": "tract_geoid", "total_geographic_records": len(tract_features), "matched_records": len(tract_features), "unmatched_records": 0, "duplicate_keys": int(tract_features["tract_geoid"].duplicated().sum()), "missing_keys": int(tract_features["tract_geoid"].isna().sum())},
    ]
    return pd.DataFrame(rows)


def build_geographic_outputs(acs_path: str | Path, places_path: str | Path, sram_path: str | Path, county_output: str | Path, tract_output: str | Path, report_output: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acs = canonicalize_acs(pd.read_csv(acs_path, dtype={"GEO_ID": "string"}))
    places = canonicalize_places(pd.read_csv(places_path, dtype={"county_fips": "string"}))
    sram = canonicalize_sram(pd.read_csv(sram_path, dtype={"tract_geoid": "string"}))

    # Outer join preserves every source geography; no values are joined by county/state name.
    county_features = places.merge(acs, on="county_fips", how="outer", suffixes=("_places", "_acs"), indicator=True, validate="one_to_one")
    county_keys = derive_county_keys(county_features["county_fips"])
    for column in county_keys:
        county_features[column] = county_keys[column]
    tract_features = sram.copy()
    report = build_join_report(acs, places, sram, county_features, tract_features)
    for path, frame in [(county_output, county_features), (tract_output, tract_features), (report_output, report)]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return county_features, tract_features, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical geographic feature tables and join report.")
    parser.add_argument("--acs-path", type=Path, default=Path("data/processed/acs_features.csv"))
    parser.add_argument("--places-path", type=Path, default=Path("data/processed/places_features.csv"))
    parser.add_argument("--sram-path", type=Path, default=Path("data/processed/sram_features.csv"))
    parser.add_argument("--county-output", type=Path, default=Path("data/processed/county_features.csv"))
    parser.add_argument("--tract-output", type=Path, default=Path("data/processed/tract_features.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("data/processed/geography_join_report.csv"))
    args = parser.parse_args()
    county, tract, report = build_geographic_outputs(**vars(args))
    print(f"Saved {len(county)} county records and {len(tract)} tract records.")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
