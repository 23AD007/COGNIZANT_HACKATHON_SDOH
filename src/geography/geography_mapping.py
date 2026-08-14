"""Canonical FIPS/GEOID normalization utilities for geographic integration."""

from __future__ import annotations

import re

import pandas as pd


def normalize_fips(values: pd.Series, width: int, key_name: str) -> pd.Series:
    """Return nullable, zero-padded FIPS strings; reject malformed non-missing values."""
    normalized = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    missing = normalized.isna() | normalized.eq("")
    invalid = ~missing & ~normalized.str.fullmatch(r"\d+")
    if invalid.any():
        examples = normalized.loc[invalid].head(3).tolist()
        raise ValueError(f"Invalid {key_name} values: {examples}")
    too_long = ~missing & normalized.str.len().gt(width)
    if too_long.any():
        examples = normalized.loc[too_long].head(3).tolist()
        raise ValueError(f"{key_name} values exceed {width} digits: {examples}")
    return normalized.mask(missing, pd.NA).str.zfill(width)


def normalize_tract_geoid(values: pd.Series) -> pd.Series:
    """Return nullable 11-digit Census tract GEOIDs."""
    return normalize_fips(values, width=11, key_name="tract_geoid")


def derive_county_keys_from_tract(tract_geoid: pd.Series) -> pd.DataFrame:
    tract = normalize_tract_geoid(tract_geoid)
    return pd.DataFrame({
        "tract_geoid": tract,
        "state_fips": tract.str[:2],
        "county_fips": tract.str[:5],
        "county_geoid": tract.str[:5],
    })


def derive_county_keys(county_fips: pd.Series) -> pd.DataFrame:
    county = normalize_fips(county_fips, width=5, key_name="county_fips")
    return pd.DataFrame({
        "state_fips": county.str[:2],
        "county_fips": county,
        "county_geoid": county,
        "tract_geoid": pd.Series(pd.NA, index=county.index, dtype="string"),
    })


def acs_geography_keys(geo_ids: pd.Series) -> pd.DataFrame:
    """Map ACS GEO_ID values only when their level explicitly includes a FIPS key."""
    geo = geo_ids.astype("string").str.strip()
    county_match = geo.str.extract(r"^0500000US(\d{5})$", expand=False)
    state_match = geo.str.extract(r"^0400000US(\d{2})$", expand=False)
    county_keys = derive_county_keys(county_match)
    county_keys["state_fips"] = county_keys["state_fips"].fillna(state_match)
    return county_keys
