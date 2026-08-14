"""County-level assembly of validated ACS, PLACES, and SRAM feature tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COUNTY_FIPS_PATTERN = r"^\d{5}$"
# S0101_C06_001E is `(X)` for every verified county and is therefore unusable downstream.
DOWNSTREAM_EXCLUDED_ACS_FEATURES = frozenset({"female_pct"})


def validate_county_fips(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Require a complete, unique five-digit county key before any merge."""
    if "county_fips" not in frame.columns:
        raise ValueError(f"{source} lacks county_fips")
    result = frame.copy()
    result["county_fips"] = result["county_fips"].astype("string")
    keys = result["county_fips"]
    if keys.isna().any() or keys.eq("").any():
        raise ValueError(f"{source} has missing county_fips")
    if not keys.str.fullmatch(COUNTY_FIPS_PATTERN, na=False).all():
        raise ValueError(f"{source} has invalid county_fips")
    if keys.duplicated().any():
        raise ValueError(f"{source} has duplicate county_fips")
    return result


def _require_no_nonkey_column_collisions(*frames: pd.DataFrame) -> None:
    seen: set[str] = set()
    for frame in frames:
        columns = set(frame.columns) - {"county_fips"}
        overlap = seen & columns
        if overlap:
            raise ValueError(f"Feature column collision across sources: {sorted(overlap)}")
        seen |= columns


def merge_county_sdoh_features(
    acs: pd.DataFrame, places: pd.DataFrame, sram: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Outer-join the three county feature sources using only ``county_fips``."""
    acs = validate_county_fips(acs, "ACS")
    places = validate_county_fips(places, "PLACES")
    sram = validate_county_fips(sram, "SRAM")
    acs = acs.drop(columns=sorted(DOWNSTREAM_EXCLUDED_ACS_FEATURES & set(acs.columns)))
    _require_no_nonkey_column_collisions(acs, places, sram)

    three_way_overlap = len(set(acs["county_fips"]) & set(places["county_fips"]) & set(sram["county_fips"]))
    merged = acs.merge(places, on="county_fips", how="outer", validate="one_to_one", indicator="_acs_places")
    merged = merged.merge(sram, on="county_fips", how="outer", validate="one_to_one", indicator="_all_sources")
    merged = merged.drop(columns=["_acs_places", "_all_sources"])
    validate_county_fips(merged, "merged county SDOH")

    missing_values = merged.isna().sum()
    return merged, {
        "rows": len(merged),
        "columns": len(merged.columns),
        "three_way_overlap": three_way_overlap,
        "missing_values": missing_values[missing_values.gt(0)].to_dict(),
    }


def build_county_sdoh_features(
    acs_path: str | Path = "data/processed/acs_features.csv",
    places_path: str | Path = "data/processed/places_features.csv",
    sram_path: str | Path = "data/processed/sram_county_features.csv",
    output_path: str | Path = "data/processed/county_sdoh_features.csv",
) -> dict[str, object]:
    """Read validated inputs and write the single requested county SDOH feature table."""
    acs = pd.read_csv(acs_path, dtype={"county_fips": "string"})
    places = pd.read_csv(places_path, dtype={"county_fips": "string"})
    sram = pd.read_csv(sram_path, dtype={"county_fips": "string"})
    merged, summary = merge_county_sdoh_features(acs, places, sram)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the county-level ACS, PLACES, and SRAM feature table.")
    parser.add_argument("--acs-path", type=Path, default=Path("data/processed/acs_features.csv"))
    parser.add_argument("--places-path", type=Path, default=Path("data/processed/places_features.csv"))
    parser.add_argument("--sram-path", type=Path, default=Path("data/processed/sram_county_features.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/county_sdoh_features.csv"))
    args = parser.parse_args()
    summary = build_county_sdoh_features(**vars(args))
    print(f"Saved {summary['rows']} counties with {summary['columns']} columns; three-way overlap: {summary['three_way_overlap']}")


if __name__ == "__main__":
    main()
