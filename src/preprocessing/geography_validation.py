"""Validation-only geographic contracts; these functions never assign geography."""

from __future__ import annotations

import re

import pandas as pd


TRACT_GEOID_PATTERN = re.compile(r"^\d{11}$")
TRACT_ACS_GEOID_PATTERN = re.compile(r"^1400000US\d{11}$")
NATIONAL_ACS_GEOID = "0100000US"
STATE_ACS_GEOID_PATTERN = re.compile(r"^0400000US\d{2}$")


class GeographicValidationError(ValueError):
    """Raised when a supplied geographic data contract is violated."""


def is_valid_tract_geoid(value: object) -> bool:
    """Return true only for an explicit 11-digit tract GEOID."""
    return isinstance(value, str) and bool(TRACT_GEOID_PATTERN.fullmatch(value.strip()))


def is_national_geoid(value: object) -> bool:
    """Return true only for the verified national ACS GEO_ID."""
    return isinstance(value, str) and value.strip() == NATIONAL_ACS_GEOID


def is_state_geoid(value: object) -> bool:
    """Return true only for a state-level ACS GEO_ID."""
    return isinstance(value, str) and bool(STATE_ACS_GEOID_PATTERN.fullmatch(value.strip()))


def validate_tract_geoid_series(series: pd.Series) -> pd.Series:
    """Return stripped valid tract GEOIDs; reject missing or non-tract values."""
    values = series.astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        raise GeographicValidationError("Missing tract_geoid values are not valid for a tract join.")
    if not values.map(is_valid_tract_geoid).all():
        raise GeographicValidationError("Invalid tract_geoid values are not valid for a tract join.")
    return values


def validate_unique_tract_geoid(frame: pd.DataFrame, column: str = "tract_geoid") -> pd.Series:
    """Require one validated tract GEOID per record; never deduplicate silently."""
    if column not in frame.columns:
        raise GeographicValidationError(f"Tract records lack {column}.")
    values = validate_tract_geoid_series(frame[column])
    if values.duplicated().any():
        raise GeographicValidationError(f"Duplicate tract IDs found in {column}.")
    return values


def validate_acs_geo_id(value: object) -> str:
    """Accept an explicit tract ACS GEO_ID and reject national/state geography."""
    geo_id = str(value).strip()
    if is_national_geoid(geo_id):
        raise GeographicValidationError("National ACS GEO_ID 0100000US is not tract-level.")
    if is_state_geoid(geo_id):
        raise GeographicValidationError(f"State ACS GEO_ID {geo_id} is not tract-level.")
    if not TRACT_ACS_GEOID_PATTERN.fullmatch(geo_id):
        raise GeographicValidationError("ACS GEO_ID must be 1400000US followed by an 11-digit tract GEOID.")
    return geo_id


def validate_member_tract_mapping(mapping: pd.DataFrame) -> None:
    """Validate a supplied mapping without creating or filling tract values."""
    required = {"member_id", "tract_geoid"}
    missing = required - set(mapping.columns)
    if missing:
        raise GeographicValidationError(f"Member-to-tract mapping missing columns: {sorted(missing)}")
    if mapping["member_id"].isna().any() or mapping["tract_geoid"].isna().any():
        raise GeographicValidationError("Member-to-tract mapping contains missing member_id or tract_geoid.")
    if mapping["member_id"].duplicated().any():
        raise GeographicValidationError("Duplicate member_id values found in member-to-tract mapping.")
    invalid = ~mapping["tract_geoid"].astype("string").map(is_valid_tract_geoid)
    if invalid.any():
        raise GeographicValidationError("Member-to-tract mapping contains invalid tract_geoid values.")


def validate_unique_tract_records(frame: pd.DataFrame, source_name: str) -> None:
    """Require complete, unique 11-digit tract keys in ACS or SRAM records."""
    try:
        validate_unique_tract_geoid(frame)
    except GeographicValidationError as error:
        raise GeographicValidationError(f"{source_name} records: {error}") from error


def validate_acs_tract_records(frame: pd.DataFrame) -> None:
    validate_unique_tract_records(frame, "ACS")


def validate_sram_tract_records(frame: pd.DataFrame) -> None:
    validate_unique_tract_records(frame, "SRAM")
