"""Create a read-only county coverage comparison; no member enrichment occurs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing.county_geography import normalize_county_fips_series


def county_overlap_report(places: pd.DataFrame, sram: pd.DataFrame) -> dict[str, int | str]:
    """Validate comparable county keys and quantify source coverage without joining."""
    for name, frame in (("PLACES", places), ("SRAM", sram)):
        if "county_fips" not in frame:
            raise ValueError(f"{name} table lacks county_fips.")
        frame["county_fips"] = normalize_county_fips_series(frame["county_fips"])
        if frame["county_fips"].isna().any():
            raise ValueError(f"{name} table contains missing county_fips.")
        if frame["county_fips"].duplicated().any():
            raise ValueError(f"{name} table contains duplicate county_fips.")

    places_keys = set(places["county_fips"])
    sram_keys = set(sram["county_fips"])
    return {
        "places_unique_counties": len(places_keys),
        "sram_unique_counties": len(sram_keys),
        "overlap_count": len(places_keys & sram_keys),
        "places_only_counties": len(places_keys - sram_keys),
        "sram_only_counties": len(sram_keys - places_keys),
        "county_acs_status": "NOT AVAILABLE: repository ACS GEO_ID values are national, not county-level.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate county SDOH source coverage without performing a merge.")
    parser.add_argument("--places-path", type=Path, default=Path("data/processed/places_features.csv"))
    parser.add_argument("--sram-path", type=Path, default=Path("data/processed/sram_county_features.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/county_sdoh_coverage_report.csv"))
    args = parser.parse_args()
    places = pd.read_csv(args.places_path, dtype={"county_fips": "string"})
    sram = pd.read_csv(args.sram_path, dtype={"county_fips": "string"})
    report = county_overlap_report(places, sram)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(args.output_path, index=False)
    print(report)


if __name__ == "__main__":
    main()
