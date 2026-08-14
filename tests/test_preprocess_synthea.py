import json

import pandas as pd

from src.preprocessing.preprocess_synthea import find_patient_table, preprocess_synthea, to_snake_case


def test_to_snake_case():
    assert to_snake_case("Health Care Expenses") == "health_care_expenses"
    assert to_snake_case(" ZIP ") == "zip"


def test_preprocess_synthea_creates_reproducible_member_file_and_report(tmp_path):
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    pd.DataFrame(
        {
            "Id": ["a", "b", "b", " "],
            "BIRTHDATE": ["2000-06-02", "not-a-date", "1980-01-01", "2010-01-01"],
            "GENDER": ["F", " M ", "M", "F"],
            "RACE": ["white", "black", "black", "asian"],
            "ETHNICITY": ["nonhispanic", "hispanic", "hispanic", "nonhispanic"],
            "MARITAL": ["S", None, None, "M"],
            "CITY": ["Boston", None, None, "Cambridge"],
            "STATE": ["Massachusetts", "Massachusetts", "Massachusetts", "Massachusetts"],
            "ZIP": ["02108", None, None, "02139"],
            "COUNTY": ["Suffolk County", None, None, "Middlesex County"],
            "FIPS": ["25025", None, None, "25017"],
            "LAT": ["42.3601", None, None, "42.3736"],
            "LON": ["-71.0589", None, None, "-71.1097"],
        }
    ).to_csv(source_dir / "patients.csv", index=False)
    output_path = tmp_path / "interim" / "members.csv"
    report_path = tmp_path / "interim" / "validation_report.json"

    cleaned, report = preprocess_synthea(source_dir, output_path, report_path, reference_date="2025-01-01")

    assert find_patient_table(source_dir).name == "patients.csv"
    assert cleaned.columns.tolist() == [
        "member_id", "birthdate", "age", "gender", "race", "ethnicity", "marital_status", "city", "state", "zip", "county", "fips", "county_fips", "lat", "lon"
    ]
    assert cleaned["member_id"].tolist() == ["a", "b"]
    assert cleaned.loc[cleaned["member_id"] == "a", "age"].item() == 24
    first_member = cleaned.loc[cleaned["member_id"] == "a"].iloc[0]
    assert first_member[["fips", "zip", "county", "lat", "lon"]].tolist() == ["25025", "02108", "Suffolk County", "42.3601", "-71.0589"]
    assert first_member["county_fips"] == "25025"
    assert pd.isna(cleaned.loc[cleaned["member_id"] == "b", "birthdate"].item())
    assert report["invalid_dates_in_source"] == 1
    assert report["duplicate_member_ids_in_source"] == 1
    assert report["rows_dropped_missing_member_id"] == 1
    assert report["geographic_coverage"]["city"]["missing_count"] == 1
    assert output_path.exists()
    assert json.loads(report_path.read_text(encoding="utf-8"))["cleaned_row_count"] == 2


def test_find_patient_table_rejects_unrelated_csv(tmp_path):
    pd.DataFrame({"encounter": ["x"], "date": ["2025-01-01"]}).to_csv(tmp_path / "encounters.csv", index=False)

    try:
        find_patient_table(tmp_path)
    except ValueError as error:
        assert "Could not identify" in str(error)
    else:
        raise AssertionError("Expected unrelated CSV to be rejected")
