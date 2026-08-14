import pandas as pd
import pytest

from src.preprocessing.geography_validation import (
    GeographicValidationError,
    is_national_geoid,
    is_state_geoid,
    is_valid_tract_geoid,
    validate_acs_geo_id,
    validate_acs_tract_records,
    validate_member_tract_mapping,
    validate_sram_tract_records,
    validate_tract_geoid_series,
    validate_unique_tract_geoid,
)


def test_valid_tract_geoid_and_invalid_national_state_geoids():
    assert is_valid_tract_geoid("01001020100")
    assert is_national_geoid("0100000US")
    assert is_state_geoid("0400000US06")
    assert validate_acs_geo_id("1400000US01001020100") == "1400000US01001020100"
    with pytest.raises(GeographicValidationError, match="National"):
        validate_acs_geo_id("0100000US")
    with pytest.raises(GeographicValidationError, match="State"):
        validate_acs_geo_id("0400000US06")


def test_missing_and_duplicate_member_tract_mappings_are_rejected():
    with pytest.raises(GeographicValidationError, match="missing"):
        validate_member_tract_mapping(pd.DataFrame({"member_id": ["m1"]}))
    with pytest.raises(GeographicValidationError, match="Duplicate member_id"):
        validate_member_tract_mapping(pd.DataFrame({"member_id": ["m1", "m1"], "tract_geoid": ["01001020100", "01001020200"]}))


def test_duplicate_acs_and_sram_tract_records_are_rejected():
    duplicated = pd.DataFrame({"tract_geoid": ["01001020100", "01001020100"]})
    with pytest.raises(GeographicValidationError, match="ACS"):
        validate_acs_tract_records(duplicated)
    with pytest.raises(GeographicValidationError, match="SRAM"):
        validate_sram_tract_records(duplicated)


def test_missing_and_duplicate_tract_ids_are_rejected():
    with pytest.raises(GeographicValidationError, match="Missing tract_geoid"):
        validate_tract_geoid_series(pd.Series([None]))
    with pytest.raises(GeographicValidationError, match="Duplicate tract IDs"):
        validate_unique_tract_geoid(pd.DataFrame({"tract_geoid": ["01001020100", "01001020100"]}))
