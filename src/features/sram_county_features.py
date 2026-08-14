"""Defensible county aggregation of repository-provided SRAM tract features."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.preprocessing.county_geography import normalize_county_fips_series


# Counts are summed. Binary tract flags are converted to counts of qualifying
# tracts. Tract percentages are deliberately excluded: source semantics do not
# establish a compatible county denominator for a defensible county rate.
SUM_COLUMNS = (
    "population_2020",
    "households_without_vehicle_count",
    "snap_households_count",
    "driving_low_access_population_beyond_1mi_10mi_count",
    "driving_no_vehicle_households_beyond_1mi_count",
    "driving_snap_households_beyond_1mi_count",
    "straight_low_access_population_beyond_1mi_10mi_count",
    "straight_no_vehicle_households_beyond_1mi_count",
    "straight_snap_households_beyond_1mi_count",
)
FLAG_COLUMNS = (
    "urban_flag",
    "low_income_tract_flag",
    "driving_low_income_low_access_flag",
    "driving_low_vehicle_access_flag",
    "straight_low_income_low_access_flag",
    "straight_low_vehicle_access_flag",
)
OMITTED_AMBIGUOUS_COLUMNS = (
    "poverty_rate_pct",
    "driving_low_access_population_beyond_1mi_pct",
    "straight_low_access_population_beyond_1mi_pct",
)


def aggregate_sram_by_county(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create one county row from tract records and return transparent coverage data."""
    required = {"tract_geoid", "county_fips", "state_fips", *SUM_COLUMNS, *FLAG_COLUMNS}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"SRAM feature table is missing required aggregation columns: {missing}")
    if features["tract_geoid"].duplicated().any():
        raise ValueError("Duplicate SRAM tract GEOIDs prevent county aggregation.")

    frame = features.copy()
    frame["county_fips"] = normalize_county_fips_series(frame["county_fips"])
    missing_county_fips = int(frame["county_fips"].isna().sum())
    assigned = frame.loc[frame["county_fips"].notna()].copy()

    # These source fields are counts; their county values are sums.
    numeric = [*SUM_COLUMNS, *FLAG_COLUMNS]
    for column in numeric:
        assigned[column] = pd.to_numeric(assigned[column], errors="coerce")
        if (assigned[column].dropna() < 0).any():
            raise ValueError(f"SRAM aggregation input contains negative values in {column}.")

    grouped = assigned.groupby("county_fips", sort=True, dropna=False)
    county = grouped.size().rename("tract_count").to_frame()
    for column in SUM_COLUMNS:
        county[f"{column}_sum"] = grouped[column].sum(min_count=1)
    for column in FLAG_COLUMNS:
        county[f"{column.removesuffix('_flag')}_tract_count"] = grouped[column].sum(min_count=1)
    county = county.reset_index()
    county["state_fips"] = county["county_fips"].str.slice(0, 2)
    ordered = ["county_fips", "state_fips", "tract_count", *[c for c in county.columns if c not in {"county_fips", "state_fips", "tract_count"}]]
    county = county[ordered]

    report = {
        "tract_rows": int(len(frame)),
        "unique_counties": int(county["county_fips"].nunique()),
        "missing_county_fips": missing_county_fips,
        "duplicate_county_level_records": int(county["county_fips"].duplicated().sum()),
        "counties_with_sram_data": int(len(county)),
        "omitted_ambiguous_columns": ";".join(OMITTED_AMBIGUOUS_COLUMNS),
        "missing_county_fips_handling": "Excluded from county table and counted in this report; not silently discarded.",
    }
    return county, report
