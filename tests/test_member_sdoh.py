import pandas as pd
import pytest

from src.features.member_sdoh import (
    validate_member_table,
    validate_county_sdoh_table,
)


def test_member_table_valid():
    df = pd.DataFrame(
        {
            "member_id": ["m1", "m2"],
            "county_fips": ["06001", "06003"],
        }
    )

    validate_member_table(df)


def test_member_table_rejects_duplicate_member():
    df = pd.DataFrame(
        {
            "member_id": ["m1", "m1"],
            "county_fips": ["06001", "06003"],
        }
    )

    with pytest.raises(ValueError):
        validate_member_table(df)


def test_county_sdoh_valid():
    df = pd.DataFrame(
        {
            "county_fips": ["06001", "06003"],
            "poverty_pct": [10.0, 15.0],
        }
    )

    validate_county_sdoh_table(df)


def test_county_sdoh_rejects_duplicate():
    df = pd.DataFrame(
        {
            "county_fips": ["06001", "06001"],
            "poverty_pct": [10.0, 15.0],
        }
    )

    with pytest.raises(ValueError):
        validate_county_sdoh_table(df)