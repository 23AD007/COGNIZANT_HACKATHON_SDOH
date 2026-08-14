"""Validated ingestion interface for future tract-level ACS data only."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TRACT_ACS_GEOID_PATTERN = r"1400000US\d{11}"


class InvalidACSTractGeographyError(ValueError):
    """Raised when ACS data is not explicitly Census-tract level."""


def validate_tract_acs_geoids(frame: pd.DataFrame) -> None:
    """Require every ACS GEO_ID to be an explicit tract GEOID; do not transform it."""
    if "GEO_ID" not in frame.columns:
        raise InvalidACSTractGeographyError("ACS tract ingestion requires a GEO_ID column.")
    geo_ids = frame["GEO_ID"].astype("string").str.strip()
    if geo_ids.isna().any() or not geo_ids.str.fullmatch(TRACT_ACS_GEOID_PATTERN, na=False).all():
        raise InvalidACSTractGeographyError(
            "ACS tract ingestion requires GEO_ID values in the form 1400000US followed by an 11-digit tract GEOID. "
            "National and state ACS GEO_ID values are rejected."
        )
    if geo_ids.duplicated().any():
        raise InvalidACSTractGeographyError("ACS tract ingestion requires unique GEO_ID values.")


def read_tract_acs(input_path: str | Path) -> pd.DataFrame:
    """Read a future supplied ACS file after verifying its tract-level geography."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Tract-level ACS input not found: {input_path}")
    frame = pd.read_csv(input_path, dtype="string")
    validate_tract_acs_geoids(frame)
    return frame
