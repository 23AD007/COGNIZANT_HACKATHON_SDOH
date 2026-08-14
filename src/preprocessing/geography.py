"""Guarded interface for a future, legitimate Synthea member-to-tract crosswalk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MEMBER_GEOGRAPHY_COLUMNS = ("member_id", "fips", "zip", "city", "state", "county", "lat", "lon")
TRACT_GEOID_PATTERN = r"\d{11}"


class MemberToTractCrosswalkRequiredError(ValueError):
    """Raised when no supplied, legitimate member-to-tract mapping is available."""


def validate_member_geography(members: pd.DataFrame) -> None:
    """Validate the retained Synthea geography interface without inferring a tract."""
    missing = set(MEMBER_GEOGRAPHY_COLUMNS) - set(members.columns)
    if missing:
        raise ValueError(f"Synthea member geography fields are missing: {sorted(missing)}")
    if members["member_id"].isna().any() or members["member_id"].duplicated().any():
        raise ValueError("member_id must be present and unique before geographic mapping.")


def load_member_to_tract_crosswalk(crosswalk_path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate a supplied member-to-tract mapping; never construct one."""
    if crosswalk_path is None or not Path(crosswalk_path).is_file():
        raise MemberToTractCrosswalkRequiredError(
            "Member-to-tract crosswalk is required. No legitimate tract mapping source was found."
        )
    crosswalk = pd.read_csv(crosswalk_path, dtype={"member_id": "string", "tract_geoid": "string"})
    required = {"member_id", "tract_geoid"}
    if not required.issubset(crosswalk.columns):
        raise MemberToTractCrosswalkRequiredError(
            "Member-to-tract crosswalk is required. No legitimate tract mapping source was found."
        )
    crosswalk["member_id"] = crosswalk["member_id"].str.strip()
    crosswalk["tract_geoid"] = crosswalk["tract_geoid"].str.strip()
    if crosswalk["member_id"].isna().any() or crosswalk["member_id"].duplicated().any():
        raise ValueError("Member-to-tract crosswalk member_id values must be present and unique.")
    if not crosswalk["tract_geoid"].str.fullmatch(TRACT_GEOID_PATTERN, na=False).all():
        raise ValueError("Member-to-tract crosswalk tract_geoid values must be valid 11-digit tract GEOIDs.")
    return crosswalk[["member_id", "tract_geoid"]]


def attach_member_tract(members: pd.DataFrame, crosswalk_path: str | Path | None = None) -> pd.DataFrame:
    """Attach only a validated external crosswalk and preserve every member."""
    validate_member_geography(members)
    crosswalk = load_member_to_tract_crosswalk(crosswalk_path)
    result = members.merge(crosswalk, on="member_id", how="left", validate="one_to_one")
    if len(result) != len(members):
        raise ValueError("Member-to-tract mapping changed the member row count.")
    return result
