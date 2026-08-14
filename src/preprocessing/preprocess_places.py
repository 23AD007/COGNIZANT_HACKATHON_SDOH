"""Normalize CDC PLACES county records for downstream community-health features."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SELECTED_MEASURE_IDS = {
    "ACCESS2", "CASTHMA", "CHECKUP", "CHD", "CHOLSCREEN", "COLON_SCREEN", "COPD",
    "CSMOKING", "DENTAL", "DIABETES", "LPA", "MAMMOUSE", "MHLTH", "OBESITY",
    "PHLTH", "STROKE",
}
REQUIRED_COLUMNS = {
    "Year", "StateAbbr", "StateDesc", "LocationName", "Measure", "Data_Value",
    "Data_Value_Unit", "Data_Value_Type", "LocationID", "MeasureId",
}
SUPPRESSION_MARKERS = {"", "-", "--", "**", "***", "(X)", "N", "NA", "N/A", "NULL", "null"}


def find_places_file(source_dir: str | Path) -> Path:
    files = list(Path(source_dir).glob("*.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one PLACES CSV in {source_dir}, found {len(files)}")
    return files[0]


def read_places_long(source_path: str | Path) -> pd.DataFrame:
    """Read selected, latest-year crude-prevalence PLACES records with safe numerics."""
    source_path = Path(source_path)
    header = pd.read_csv(source_path, nrows=0)
    missing = REQUIRED_COLUMNS - set(header.columns)
    if missing:
        raise ValueError(f"PLACES file missing required columns: {sorted(missing)}")
    frame = pd.read_csv(source_path, usecols=sorted(REQUIRED_COLUMNS), dtype="string", low_memory=False)
    frame["Year"] = pd.to_numeric(frame["Year"], errors="coerce")
    latest_year = int(frame["Year"].max())
    frame = frame.loc[
        (frame["Year"] == latest_year)
        & (frame["Data_Value_Type"].str.strip() == "Crude prevalence")
        & frame["MeasureId"].isin(SELECTED_MEASURE_IDS)
    ].copy()
    if frame.empty:
        raise ValueError("No selected PLACES records found for the latest year and crude prevalence.")

    frame["county_fips"] = frame["LocationID"].str.strip()
    # The county extract includes one U.S. aggregate record (LocationID "59").
    # Keep only explicit five-digit county FIPS identifiers; never manufacture a key.
    frame = frame.loc[frame["county_fips"].str.fullmatch(r"\d{5}", na=False)].copy()
    if frame.empty:
        raise ValueError("No five-digit county FIPS records remain in the PLACES extract.")
    numeric_text = frame["Data_Value"].str.strip().replace(list(SUPPRESSION_MARKERS), pd.NA)
    frame["prevalence_pct"] = pd.to_numeric(numeric_text, errors="coerce")
    invalid = frame["prevalence_pct"].notna() & ~frame["prevalence_pct"].between(0, 100)
    if invalid.any():
        raise ValueError("PLACES prevalence values must be between 0 and 100.")

    long = frame.rename(columns={
        "StateAbbr": "state_abbr", "StateDesc": "state_name", "LocationName": "county_name",
        "MeasureId": "measure_id", "Measure": "measure_name", "Data_Value_Unit": "value_unit",
        "Data_Value_Type": "prevalence_type", "Year": "places_year",
    })[["county_fips", "county_name", "state_abbr", "state_name", "places_year", "prevalence_type", "measure_id", "measure_name", "value_unit", "prevalence_pct"]]
    if long.duplicated(["county_fips", "places_year", "prevalence_type", "measure_id"]).any():
        raise ValueError("Duplicate county/measure PLACES records detected.")
    return long


def preprocess_places(source_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    long = read_places_long(find_places_file(source_dir))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(output_path, index=False)
    return long


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize CDC PLACES county data.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/places"))
    parser.add_argument("--output-path", type=Path, default=Path("data/interim/places_clean/places_long.csv"))
    args = parser.parse_args()
    long = preprocess_places(args.source_dir, args.output_path)
    print(f"Saved {len(long)} PLACES records across {long['county_fips'].nunique()} counties to {args.output_path}")


if __name__ == "__main__":
    main()
