"""County geographic-key validation for the county-first SDOH pipeline.

This module performs format validation only.  A five-digit value is not
asserted to be an extant county without an authoritative county reference
dataset; it is simply normalized for safe key comparison.
"""

from __future__ import annotations

import re
from numbers import Integral, Real
from typing import Any

import pandas as pd


COUNTY_FIPS_PATTERN = re.compile(r"^\d{5}$")
COUNTY_ACS_GEOID_PATTERN = re.compile(r"^0500000US(\d{5})$")


class CountyGeographyError(ValueError):
    """Raised when a purported county geographic key has an invalid format."""


def normalize_county_fips(value: Any) -> str | pd._libs.missing.NAType:
    """Normalize a county FIPS value to five digits without creating geography.

    Missing inputs remain missing. Integer-like values with one to five digits
    are left-padded, which preserves a valid leading-zero representation. The
    function rejects non-numeric, fractional, zero, and over-length inputs. It
    does not validate that the resulting code exists in a county reference file.
    """
    if value is None or pd.isna(value):
        return pd.NA

    if isinstance(value, bool):
        raise CountyGeographyError("County FIPS must be a numeric identifier, not a boolean.")
    if isinstance(value, Integral):
        text = str(value)
    elif isinstance(value, Real):
        if not float(value).is_integer():
            raise CountyGeographyError(f"County FIPS must be integer-like: {value!r}")
        text = str(int(value))
    else:
        text = str(value).strip()
        if not text:
            return pd.NA
        if re.fullmatch(r"\d+\.0+", text):
            text = text.split(".", maxsplit=1)[0]

    if not text.isdigit():
        raise CountyGeographyError(f"County FIPS must contain only digits: {value!r}")
    if len(text) > 5 or int(text) == 0:
        raise CountyGeographyError(f"County FIPS must be a non-zero value of at most five digits: {value!r}")

    normalized = text.zfill(5)
    if not COUNTY_FIPS_PATTERN.fullmatch(normalized):  # defensive; should be unreachable
        raise CountyGeographyError(f"County FIPS could not be normalized: {value!r}")
    return normalized


def normalize_county_fips_series(values: pd.Series) -> pd.Series:
    """Normalize a Series of county FIPS values while retaining nullable strings."""
    return values.map(normalize_county_fips).astype("string")


def validate_county_acs_geo_id(value: Any) -> str:
    """Require a Census county-level ACS GEO_ID (``0500000US`` + 5 digits)."""
    if value is None or pd.isna(value) or not str(value).strip():
        raise CountyGeographyError("County ACS GEO_ID is missing.")
    text = str(value).strip()
    if text == "0100000US":
        raise CountyGeographyError("National ACS GEO_ID cannot be used as county data.")
    if re.fullmatch(r"0400000US\d{2}", text):
        raise CountyGeographyError("State ACS GEO_ID cannot be used as county data.")
    if not COUNTY_ACS_GEOID_PATTERN.fullmatch(text):
        raise CountyGeographyError(
            "County ACS GEO_ID must use the county format 0500000US followed by a five-digit county FIPS."
        )
    return text
