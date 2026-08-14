"""Read-only guard against invalid geographic inputs for the SDOH pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

try:
    from src.preprocessing.geography import MEMBER_GEOGRAPHY_COLUMNS, load_member_to_tract_crosswalk
    from src.preprocessing.geography_validation import TRACT_ACS_GEOID_PATTERN, validate_unique_tract_geoid
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.preprocessing.geography import MEMBER_GEOGRAPHY_COLUMNS, load_member_to_tract_crosswalk
    from src.preprocessing.geography_validation import TRACT_ACS_GEOID_PATTERN, validate_unique_tract_geoid


ACS_TABLE_TOKENS = ("DP02", "DP03", "DP04", "DP05", "S0101", "S1501", "S1701", "S1901")
SYNTHEA_SOURCE_COLUMNS = {"Id", "FIPS", "ZIP", "CITY", "STATE", "COUNTY", "LAT", "LON"}


def synthea_member_geography_available(patients_path: Path) -> bool:
    if not patients_path.is_file():
        return False
    header = set(pd.read_csv(patients_path, nrows=0).columns)
    return SYNTHEA_SOURCE_COLUMNS.issubset(header)


def tract_acs_available(acs_dir: Path) -> bool:
    """Require every configured ACS table to contain only actual tract GEO_ID records."""
    for token in ACS_TABLE_TOKENS:
        matches = list(acs_dir.glob(f"*{token}-Data.csv"))
        if len(matches) != 1:
            return False
        with matches[0].open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "GEO_ID" not in (reader.fieldnames or []):
                return False
            geo_ids = [(row.get("GEO_ID") or "").strip() for row in reader]
        # Census exports include a descriptive 'Geography' row; it is not a geographic record.
        records = [geo_id for geo_id in geo_ids if geo_id != "Geography"]
        if not records or not all(TRACT_ACS_GEOID_PATTERN.fullmatch(geo_id) for geo_id in records):
            return False
    return True


def member_tract_crosswalk_available(crosswalk_path: Path | None) -> bool:
    try:
        load_member_to_tract_crosswalk(crosswalk_path)
    except (FileNotFoundError, ValueError):
        return False
    return True


def sram_tract_status(path: Path) -> tuple[bool, int, int, int]:
    if not path.is_file():
        return False, 0, 0, 0
    frame = pd.read_csv(path, dtype={"tract_geoid": "string"})
    missing_values = int(frame.isna().sum().sum())
    try:
        keys = validate_unique_tract_geoid(frame)
    except ValueError:
        return False, len(frame), int(frame.get("tract_geoid", pd.Series(dtype="string")).isna().sum()), missing_values
    return True, len(frame), int(keys.isna().sum()), missing_values


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Report read-only SDOH geographic readiness.")
    parser.add_argument("--patients-path", type=Path, default=Path("data/raw/synthea/patients.csv"))
    parser.add_argument("--acs-dir", type=Path, default=Path("data/raw/acs"))
    parser.add_argument("--member-tract-crosswalk", type=Path, default=None)
    parser.add_argument("--sram-features-path", type=Path, default=Path("data/processed/sram_features.csv"))
    parser.add_argument("--tract-features-path", type=Path, default=Path("data/processed/tract_features.csv"))
    args = parser.parse_args()

    synthea_available = synthea_member_geography_available(args.patients_path)
    acs_available = tract_acs_available(args.acs_dir)
    crosswalk_available = member_tract_crosswalk_available(args.member_tract_crosswalk)
    sram_available, sram_rows, sram_missing_keys, sram_missing_values = sram_tract_status(args.sram_features_path)
    tract_available, tract_rows, tract_missing_keys, tract_missing_values = sram_tract_status(args.tract_features_path)
    ready = synthea_available and acs_available and crosswalk_available and sram_available and tract_available

    print("========================================")
    print("SDOH GEOGRAPHY READINESS")
    print("========================================")
    print(f"Synthea member geography: {'AVAILABLE' if synthea_available else 'NOT AVAILABLE'}")
    print(f"Member tract mapping: {'AVAILABLE' if crosswalk_available else 'NOT AVAILABLE'}")
    print(f"Tract-level ACS: {'AVAILABLE' if acs_available else 'NOT AVAILABLE'}")
    print(f"Tract-level SRAM: {'AVAILABLE' if sram_available and tract_available else 'NOT AVAILABLE'}")
    print(f"SRAM feature rows: {sram_rows}; missing tract keys: {sram_missing_keys}; missing values: {sram_missing_values}")
    print(f"Tract feature rows: {tract_rows}; missing tract keys: {tract_missing_keys}; missing values: {tract_missing_values}")
    print("----------------------------------------")
    print(f"FINAL STATUS: {'READY' if ready else 'BLOCKED'}")
    print("----------------------------------------")
    if not ready:
        print("BLOCKERS:")
        if not crosswalk_available:
            print("1. Missing member → tract mapping")
        if not acs_available:
            print("2. Missing tract-level ACS")


if __name__ == "__main__":
    main()
