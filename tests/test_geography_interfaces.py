import pandas as pd
import pytest

from src.preprocessing.acs_tract_ingestion import InvalidACSTractGeographyError, validate_tract_acs_geoids
from src.preprocessing.geography import MemberToTractCrosswalkRequiredError, load_member_to_tract_crosswalk, validate_member_geography


def test_national_acs_geoid_is_rejected():
    with pytest.raises(InvalidACSTractGeographyError, match="National and state"):
        validate_tract_acs_geoids(pd.DataFrame({"GEO_ID": ["0100000US"]}))


def test_state_acs_geoid_is_rejected():
    with pytest.raises(InvalidACSTractGeographyError, match="National and state"):
        validate_tract_acs_geoids(pd.DataFrame({"GEO_ID": ["0400000US06"]}))


def test_valid_tract_form_acs_geoid_is_accepted():
    validate_tract_acs_geoids(pd.DataFrame({"GEO_ID": ["1400000US01001020100"]}))


def test_missing_member_to_tract_crosswalk_is_blocked():
    with pytest.raises(MemberToTractCrosswalkRequiredError, match="Member-to-tract crosswalk is required"):
        load_member_to_tract_crosswalk()


def test_synthea_geography_fields_are_preserved_without_deriving_tract():
    members = pd.DataFrame({
        "member_id": ["member-1"], "fips": ["25025"], "zip": ["02124"], "city": ["Boston"],
        "state": ["Massachusetts"], "county": ["Suffolk County"], "lat": ["42.354248842342805"], "lon": ["-71.06877841138558"],
    })
    validate_member_geography(members)
    assert members.loc[0, ["fips", "zip", "city", "state", "county", "lat", "lon"]].tolist() == [
        "25025", "02124", "Boston", "Massachusetts", "Suffolk County", "42.354248842342805", "-71.06877841138558"
    ]
