import pandas as pd
import pytest

from src.features.places_features import build_places_features
from src.preprocessing.preprocess_places import read_places_long


def _source_rows(values):
    return pd.DataFrame({
        "Year": ["2023"] * len(values), "StateAbbr": ["MA"] * len(values), "StateDesc": ["Massachusetts"] * len(values),
        "LocationName": ["Suffolk"] * len(values), "Measure": ["Diagnosed diabetes among adults"] * len(values),
        "Data_Value": values, "Data_Value_Unit": ["%"] * len(values), "Data_Value_Type": ["Crude prevalence"] * len(values),
        "LocationID": ["25025"] * len(values), "MeasureId": ["DIABETES"] * len(values),
    })


def test_places_geographic_key_and_missing_value_handling(tmp_path):
    path = tmp_path / "places.csv"
    _source_rows(["(X)"]).to_csv(path, index=False)
    long = read_places_long(path)
    assert long["county_fips"].tolist() == ["25025"]
    assert pd.isna(long["prevalence_pct"].item())


def test_duplicate_county_measure_is_detected(tmp_path):
    path = tmp_path / "places.csv"
    _source_rows(["10", "11"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate county/measure"):
        read_places_long(path)


def test_feature_values_are_numeric_and_in_prevalence_range():
    long = pd.DataFrame({
        "county_fips": ["25025", "25025"], "county_name": ["Suffolk", "Suffolk"], "state_abbr": ["MA", "MA"],
        "state_name": ["Massachusetts", "Massachusetts"], "places_year": [2023, 2023], "prevalence_type": ["Crude prevalence", "Crude prevalence"],
        "measure_id": ["DIABETES", "OBESITY"], "prevalence_pct": [11.2, 29.5],
    })
    features = build_places_features(long)
    assert features["county_fips"].is_unique
    assert pd.api.types.is_numeric_dtype(features["places_diabetes_pct"])
    assert features.filter(like="places_").drop(columns=["places_year"], errors="ignore").stack().between(0, 100).all()
