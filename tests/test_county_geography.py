import pandas as pd
import pytest

from src.features.sram_county_features import aggregate_sram_by_county
from src.preprocessing.county_geography import CountyGeographyError, normalize_county_fips, validate_county_acs_geo_id
from scripts.check_county_sdoh_coverage import county_overlap_report


def test_county_fips_normalization_preserves_leading_zeroes():
    assert normalize_county_fips("25025") == "25025"
    assert normalize_county_fips("05043") == "05043"
    assert normalize_county_fips("6001") == "06001"


def test_county_fips_missing_and_invalid_values():
    assert pd.isna(normalize_county_fips(None))
    assert pd.isna(normalize_county_fips(" "))
    for value in ("abc", "123456", "12.5", 0, True):
        with pytest.raises(CountyGeographyError):
            normalize_county_fips(value)


def test_county_acs_validation_rejects_national_and_state_geoids():
    assert validate_county_acs_geo_id("0500000US25025") == "0500000US25025"
    with pytest.raises(CountyGeographyError, match="National"):
        validate_county_acs_geo_id("0100000US")
    with pytest.raises(CountyGeographyError, match="State"):
        validate_county_acs_geo_id("0400000US06")
    with pytest.raises(CountyGeographyError, match="county format"):
        validate_county_acs_geo_id("1400000US25025010100")


def _sram_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "tract_geoid": ["01001020100", "01001020200", "01003010100"],
        "county_fips": ["01001", "1001", "01003"], "state_fips": ["01", "01", "01"],
        "population_2020": [100, 200, 50], "households_without_vehicle_count": [2, 3, 1],
        "snap_households_count": [4, 5, 2],
        "driving_low_access_population_beyond_1mi_10mi_count": [10, 20, 5],
        "driving_no_vehicle_households_beyond_1mi_count": [1, 2, 1],
        "driving_snap_households_beyond_1mi_count": [3, 4, 1],
        "straight_low_access_population_beyond_1mi_10mi_count": [7, 8, 2],
        "straight_no_vehicle_households_beyond_1mi_count": [1, 1, 0],
        "straight_snap_households_beyond_1mi_count": [2, 3, 1],
        "urban_flag": [1, 1, 0], "low_income_tract_flag": [0, 1, 1],
        "driving_low_income_low_access_flag": [0, 1, 1], "driving_low_vehicle_access_flag": [0, 0, 1],
        "straight_low_income_low_access_flag": [0, 1, 0], "straight_low_vehicle_access_flag": [0, 1, 1],
    })


def test_sram_county_aggregation_sums_counts_and_counts_flagged_tracts():
    county, report = aggregate_sram_by_county(_sram_rows())
    autauga = county.loc[county["county_fips"] == "01001"].iloc[0]
    assert autauga["tract_count"] == 2
    assert autauga["population_2020_sum"] == 300
    assert autauga["low_income_tract_tract_count"] == 1
    assert report["missing_county_fips"] == 0
    assert "poverty_rate_pct" in report["omitted_ambiguous_columns"]


def test_sram_missing_county_fips_is_reported_not_silently_hidden():
    rows = _sram_rows()
    rows.loc[0, "county_fips"] = pd.NA
    county, report = aggregate_sram_by_county(rows)
    assert report["missing_county_fips"] == 1
    assert len(county) == 2


def test_places_and_sram_county_keys_are_normalized_before_coverage_comparison():
    report = county_overlap_report(
        pd.DataFrame({"county_fips": ["6001", "25025"]}),
        pd.DataFrame({"county_fips": ["06001", "25025"]}),
    )
    assert report["overlap_count"] == 2
    assert report["places_only_counties"] == 0
