import pandas as pd
import pytest

from src.synthetic.generate_members import generate_members


def _member_rows():
    return pd.DataFrame({"member_id": ["m1", "m2"], "birthdate": ["1980-01-01", "1990-01-01"], "age": [45, 35], "gender": ["F", "M"], "race": ["white", "black"], "ethnicity": ["nonhispanic", "hispanic"], "marital_status": ["M", "S"], "city": ["Boston", "Boston"], "state": ["Massachusetts", "Massachusetts"], "zip": ["02108", "02109"]})


def test_generation_preserves_members_and_does_not_assign_tracts(tmp_path):
    members, raw, county, output = (tmp_path / name for name in ("members.csv", "patients.csv", "county.csv", "output.csv"))
    _member_rows().to_csv(members, index=False)
    pd.DataFrame({"Id": ["m1", "m2"], "FIPS": ["25025", None]}).to_csv(raw, index=False)
    pd.DataFrame({"county_fips": ["25025"], "places_diabetes_pct": [11.2]}).to_csv(county, index=False)
    generated = generate_members(members, raw, county, output)
    assert generated["member_id"].is_unique and len(generated) == 2
    assert generated["county_fips"].tolist() == ["25025", pd.NA]
    assert generated["county_context_available"].tolist() == [True, False]
    assert generated["tract_geoid"].isna().all()
    assert not generated["tract_context_available"].any()
    assert generated["places_diabetes_pct"].iloc[0] == 11.2


def test_duplicate_county_context_is_rejected(tmp_path):
    members, raw, county, output = (tmp_path / name for name in ("members.csv", "patients.csv", "county.csv", "output.csv"))
    _member_rows().to_csv(members, index=False)
    pd.DataFrame({"Id": ["m1", "m2"], "FIPS": ["25025", "25025"]}).to_csv(raw, index=False)
    pd.DataFrame({"county_fips": ["25025", "25025"], "places_diabetes_pct": [11.2, 12.0]}).to_csv(county, index=False)
    with pytest.raises(ValueError, match="duplicate county_fips"):
        generate_members(members, raw, county, output)
