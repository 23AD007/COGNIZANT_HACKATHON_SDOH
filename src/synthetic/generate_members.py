"""Create member-level synthetic records with available geographic context only."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from src.geography.geography_mapping import normalize_fips
except ModuleNotFoundError:  # Enables direct script execution.
    from geography_mapping import normalize_fips


BASE_COLUMNS = ["member_id", "birthdate", "age", "gender", "race", "ethnicity", "marital_status", "city", "state", "zip"]


def load_member_counties(raw_patients_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_patients_path, usecols=["Id", "FIPS"], dtype="string")
    raw = raw.rename(columns={"Id": "member_id", "FIPS": "county_fips"})
    raw["member_id"] = raw["member_id"].str.strip().replace("", pd.NA)
    if raw["member_id"].isna().any() or raw["member_id"].duplicated().any():
        raise ValueError("Raw Synthea patient IDs must be present and unique.")
    raw["county_fips"] = normalize_fips(raw["county_fips"], width=5, key_name="Synthea county_fips")
    return raw


def prepare_county_context(county_features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "county_fips" not in county_features:
        raise ValueError("County features lack county_fips.")
    context = county_features.loc[county_features["county_fips"].notna()].copy()
    context["county_fips"] = normalize_fips(context["county_fips"], width=5, key_name="county_fips")
    if context["county_fips"].duplicated().any():
        raise ValueError("County features contain duplicate county_fips values.")
    # PLACES fields are the only available county context in this source bundle.
    places_columns = [column for column in context if column.startswith("places_")]
    return context[["county_fips", *places_columns]], places_columns


def validate_members(members: pd.DataFrame, expected_rows: int) -> None:
    if len(members) != expected_rows:
        raise ValueError("A geographic join changed the member row count.")
    if members["member_id"].isna().any() or members["member_id"].duplicated().any():
        raise ValueError("member_id must be present and unique.")
    invalid_age = members["age"].notna() & ~members["age"].between(0, 120)
    if invalid_age.any():
        raise ValueError("Member ages must be between 0 and 120 when available.")
    valid_county = members["county_fips"].isna() | members["county_fips"].str.fullmatch(r"\d{5}", na=False)
    valid_tract = members["tract_geoid"].isna() | members["tract_geoid"].str.fullmatch(r"\d{11}", na=False)
    if not valid_county.all() or not valid_tract.all():
        raise ValueError("Member geographic keys are invalid.")


def generate_members(members_path: str | Path, raw_patients_path: str | Path, county_features_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    members = pd.read_csv(members_path, dtype={"member_id": "string", "zip": "string"})
    missing = set(BASE_COLUMNS) - set(members.columns)
    if missing:
        raise ValueError(f"Cleaned Synthea members lack fields: {sorted(missing)}")
    members = members[BASE_COLUMNS].copy()
    members["member_id"] = members["member_id"].astype("string").str.strip()
    members["age"] = pd.to_numeric(members["age"], errors="coerce").astype("Int64")
    counties = load_member_counties(raw_patients_path)
    context, places_columns = prepare_county_context(pd.read_csv(county_features_path, dtype={"county_fips": "string"}))
    expected_rows = len(members)
    enriched = members.merge(counties, on="member_id", how="left", validate="one_to_one")
    enriched = enriched.merge(context, on="county_fips", how="left", validate="many_to_one", indicator="_county_merge")
    enriched["county_context_available"] = enriched["_county_merge"].eq("both")
    # ACS has no county-level feature row in the supplied source, so do not apply its national values.
    enriched["acs_county_context_available"] = False
    # Synthea has no tract GEOID and no geographic crosswalk was supplied; no tract is assigned.
    enriched["tract_geoid"] = pd.Series(pd.NA, index=enriched.index, dtype="string")
    enriched["tract_context_available"] = False
    enriched = enriched.drop(columns="_county_merge")
    validate_members(enriched, expected_rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Synthea members with available geographic context.")
    parser.add_argument("--members-path", type=Path, default=Path("data/interim/synthea_clean/members.csv"))
    parser.add_argument("--raw-patients-path", type=Path, default=Path("data/raw/synthea/patients.csv"))
    parser.add_argument("--county-features-path", type=Path, default=Path("data/processed/county_features.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/synthetic_members_base.csv"))
    args = parser.parse_args()
    members = generate_members(**vars(args))
    print(f"Saved {len(members)} unique synthetic members to {args.output_path}")
    print(f"County context available: {int(members['county_context_available'].sum())}; tract context available: {int(members['tract_context_available'].sum())}")


if __name__ == "__main__":
    main()
