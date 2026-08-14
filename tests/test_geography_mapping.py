import pandas as pd
import pytest

from src.geography.geographic_merge import assert_unique, canonicalize_places
from src.geography.geography_mapping import derive_county_keys, normalize_fips, normalize_tract_geoid


def test_fips_and_geoid_normalization_preserves_leading_zeros():
    assert normalize_fips(pd.Series(["1", "25"]), 2, "state_fips").tolist() == ["01", "25"]
    county = derive_county_keys(pd.Series(["1001", "25025"]))
    assert county["county_fips"].tolist() == ["01001", "25025"]
    assert normalize_tract_geoid(pd.Series(["1001020100"])).item() == "01001020100"


def test_duplicate_geographic_keys_are_detected():
    frame = pd.DataFrame({"county_fips": ["25025", "25025"]})
    with pytest.raises(ValueError, match="duplicate county_fips"):
        assert_unique(frame, "county_fips", "test")
    with pytest.raises(ValueError, match="duplicate county_fips"):
        canonicalize_places(pd.DataFrame({"county_fips": ["25025", "25025"]}))
