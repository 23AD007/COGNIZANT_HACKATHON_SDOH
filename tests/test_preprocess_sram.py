import pandas as pd
import pytest

from src.features.sram_features import build_sram_features
from src.preprocessing.preprocess_sram import normalize_tract_geoid


def _components():
    values = {"tract_geoid": ["01001020100"], "state_fips": ["01"], "county_fips": ["01001"], "tract_code": ["020100"], "State": ["Alabama"], "County20": ["Autauga County"]}
    for column, value in {"Urban": 1, "POP2020": 1775, "PovertyRate": 17.4, "LowIncomeTracts": 0, "TractHUNV": 35, "TractSNAP": 100, "DD_SRAM_LILATracts_1And10": 0, "DD_SRAM_HUNVFlag": 0, "DD_SRAM_LAPOP1_10": 200, "DD_SRAM_lapop1share": 12.5, "DD_SRAM_lahunv1": 5, "DD_SRAM_lasnap1": 15, "SD_SRAM_LILATracts_1And10": 0, "SD_SRAM_HUNVFlag": 0, "SD_SRAM_LAPOP1_10": 150, "SD_SRAM_lapop1share": 9.5, "SD_SRAM_lahunv1": 3, "SD_SRAM_lasnap1": 10}.items(): values[column] = [value]
    return pd.DataFrame(values)


def test_normalize_tract_geoid_and_final_key_structure():
    assert normalize_tract_geoid(pd.Series(["1001020100"])).item() == "01001020100"
    features = build_sram_features(_components())
    assert features["tract_geoid"].str.fullmatch(r"\d{11}").all()


def test_duplicate_tract_keys_are_detected():
    components = pd.concat([_components(), _components()], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate tract GEOIDs"):
        build_sram_features(components)


def test_distance_and_percentage_ranges_are_validated():
    features = build_sram_features(_components())
    assert (features.filter(regex="driving_|straight_") >= 0).all().all()
    assert features[["poverty_rate_pct", "driving_low_access_population_beyond_1mi_pct", "straight_low_access_population_beyond_1mi_pct"]].stack().between(0, 100).all()
    invalid = _components()
    invalid["DD_SRAM_lapop1share"] = 101
    with pytest.raises(ValueError, match="percentage/rate"):
        build_sram_features(invalid)
