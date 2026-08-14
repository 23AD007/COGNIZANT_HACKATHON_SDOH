import pandas as pd
import pytest

from src.features.county_sdoh_features import merge_county_sdoh_features


def test_county_sdoh_merge_uses_county_fips_and_preserves_unmatched_counties():
    acs = pd.DataFrame({"county_fips": ["01001", "01003"], "acs_feature": [1, 2], "female_pct": [pd.NA, pd.NA]})
    places = pd.DataFrame({"county_fips": ["01003", "01005"], "places_feature": [3, 4]})
    sram = pd.DataFrame({"county_fips": ["01001", "01003"], "sram_feature": [5, 6]})

    merged, summary = merge_county_sdoh_features(acs, places, sram)

    assert merged["county_fips"].tolist() == ["01001", "01003", "01005"]
    assert summary["three_way_overlap"] == 1
    assert summary["rows"] == 3
    assert summary["columns"] == 4
    assert summary["missing_values"] == {"acs_feature": 1, "places_feature": 1, "sram_feature": 1}
    assert "female_pct" not in merged.columns
    assert merged.loc[merged["county_fips"].eq("01003"), "acs_feature"].item() == 2


@pytest.mark.parametrize(
    "source, values, error",
    [
        ("ACS", ["01001", "01001"], "duplicate county_fips"),
        ("PLACES", ["1001"], "invalid county_fips"),
        ("SRAM", [None], "missing county_fips"),
    ],
)
def test_county_sdoh_merge_rejects_invalid_keys(source, values, error):
    frames = {
        "ACS": pd.DataFrame({"county_fips": ["01001"], "acs_feature": [1]}),
        "PLACES": pd.DataFrame({"county_fips": ["01001"], "places_feature": [1]}),
        "SRAM": pd.DataFrame({"county_fips": ["01001"], "sram_feature": [1]}),
    }
    frames[source] = pd.DataFrame({"county_fips": values, f"{source.lower()}_feature": range(len(values))})
    with pytest.raises(ValueError, match=error):
        merge_county_sdoh_features(frames["ACS"], frames["PLACES"], frames["SRAM"])
