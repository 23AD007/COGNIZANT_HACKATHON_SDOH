import pandas as pd
import pytest

from src.preprocessing.preprocess_acs import COMPONENTS, TABLES, read_and_clean_table, selected_components


def test_clean_table_preserves_geo_and_excludes_margin_of_error(tmp_path):
    data = tmp_path / "DP02-Data.csv"
    metadata = tmp_path / "DP02-Column-Metadata.csv"
    pd.DataFrame({"GEO_ID": ["Geography", "0100000US"], "NAME": ["Geographic Area Name", "United States"], "DP02_0060PE": ["label", "4.2"], "DP02_0061PE": ["label", "5.1"], "DP02_0153PE": ["label", "90"], "DP02_0154PE": ["label", "80"], "DP02_0060PM": ["label", "0.2"]}).to_csv(data, index=False)
    pd.DataFrame({"Column Name": ["DP02_0060PE", "DP02_0061PE", "DP02_0153PE", "DP02_0154PE", "DP02_0060PM"], "Label": ["Percent!!Educational attainment!!Less than 9th grade", "Percent!!Educational attainment!!9th to 12th grade, no diploma", "Percent!!Computers!!With a computer", "Percent!!Computers!!With broadband", "Percent Margin of Error!!Educational attainment!!Less than 9th grade"]}).to_csv(metadata, index=False)
    cleaned, mapping = read_and_clean_table(data, metadata, "DP02")
    assert cleaned["GEO_ID"].tolist() == ["0100000US"]
    assert cleaned["geographic_level"].tolist() == ["national"]
    assert cleaned["education_less_than_9th_pct"].tolist() == [4.2]
    assert "DP02_0060PM" not in cleaned.columns
    assert mapping["education_less_than_9th_pct"]["metadata_label"].startswith("Percent")


def test_duplicate_geographic_keys_are_detected(tmp_path):
    data = tmp_path / "DP02-Data.csv"
    metadata = tmp_path / "DP02-Column-Metadata.csv"
    pd.DataFrame({"GEO_ID": ["0100000US", "0100000US"], "NAME": ["United States", "United States"], "DP02_0060PE": ["4", "4"]}).to_csv(data, index=False)
    pd.DataFrame({"Column Name": ["DP02_0060PE"], "Label": ["Percent!!Educational attainment!!Less than 9th grade"]}).to_csv(metadata, index=False)
    with pytest.raises(ValueError, match="duplicate GEO_ID"):
        read_and_clean_table(data, metadata, "DP02")


def test_current_component_selection_uses_only_documented_non_moe_features():
    selected = {table: selected_components(table) for table in TABLES}
    assert set().union(*[set(components) for components in selected.values()]) == set(COMPONENTS)
    assert selected["DP02"] == {
        "education_less_than_9th_pct": "DP02_0060PE",
        "education_9th_to_12th_no_diploma_pct": "DP02_0061PE",
        "digital_with_computer_pct": "DP02_0153PE",
        "digital_with_broadband_pct": "DP02_0154PE",
    }
    assert not any(code.endswith(("M", "PM")) for components in selected.values() for code in components.values())
