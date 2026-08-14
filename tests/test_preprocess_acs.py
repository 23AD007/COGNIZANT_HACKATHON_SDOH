import pandas as pd
import pytest

from src.features.acs_features import build_acs_features
from src.preprocessing.preprocess_acs import TABLES, read_and_clean_table


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


def test_feature_table_has_numeric_features_and_no_moe_columns(tmp_path):
    base = {"GEO_ID": ["0100000US"], "NAME": ["United States"], "geographic_level": ["national"]}
    for table in TABLES:
        pd.DataFrame(base).to_csv(tmp_path / f"{table.lower()}_clean.csv", index=False)
    values = {
        "dp02_clean.csv": {"education_less_than_9th_pct": 4, "education_9th_to_12th_no_diploma_pct": 6, "digital_with_computer_pct": 90, "digital_with_broadband_pct": 80},
        "dp03_clean.csv": {"unemployment_pct": 5, "public_assistance_pct": 2, "uninsured_pct": 8, "mean_commute_minutes": 25},
        "dp04_clean.csv": {"housing_vacancy_pct": 10, "housing_no_vehicle_pct": 7, "housing_renter_pct": 35, "housing_crowded_1_01_to_1_50_pct": 2, "housing_crowded_1_51_plus_pct": 1, "housing_rent_30_to_34_9_pct": 15, "housing_rent_35_plus_pct": 20},
        "s0101_clean.csv": {"population_total": 1000, "median_age_years": 39, "female_pct": 51},
        "s1501_clean.csv": {"education_high_school_pct": 30, "education_college_pct": 20, "education_bachelors_or_higher_pct": 25},
        "s1701_clean.csv": {"poverty_pct": 12},
        "s1901_clean.csv": {"median_household_income": 70000},
    }
    for name, columns in values.items():
        path = tmp_path / name
        frame = pd.read_csv(path)
        for column, value in columns.items(): frame[column] = value
        frame.to_csv(path, index=False)
    features = build_acs_features(tmp_path)
    assert "GEO_ID" in features and features["GEO_ID"].is_unique
    assert features["digital_no_computer_pct"].item() == 10
    assert features["housing_cost_burden_30pct_or_more"].item() == 35
    assert not any("margin" in column.lower() or column.lower().endswith("moe") for column in features.columns)
    assert pd.api.types.is_numeric_dtype(features["poverty_pct"])
