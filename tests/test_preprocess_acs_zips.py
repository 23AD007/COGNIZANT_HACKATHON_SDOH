from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.preprocess_acs import (
    COMPONENTS,
    EXPECTED_COUNTY_ROWS,
    TABLES,
    discover_zip_inputs,
    preprocess_acs,
    validate_county_geoids,
    zip_members,
)


RAW_ACS_DIR = Path("data/raw/acs")


def _require_real_zips() -> dict[str, Path]:
    try:
        return discover_zip_inputs(RAW_ACS_DIR)
    except FileNotFoundError as exc:
        pytest.skip(f"Real ACS ZIP inputs are unavailable: {exc}")


def test_real_zip_inputs_are_intact_and_contain_required_members():
    for table, source_zip in _require_real_zips().items():
        data, metadata = zip_members(source_zip, table)
        assert data.filename.endswith(f"{table}-Data.csv")
        assert metadata.filename.endswith(f"{table}-Column-Metadata.csv")


def test_county_geoid_validation_rejects_duplicate_and_missing_keys():
    valid = "0500000US01001"
    with pytest.raises(ValueError, match="duplicate GEO_ID"):
        validate_county_geoids(pd.DataFrame({"GEO_ID": [valid] * EXPECTED_COUNTY_ROWS}), "DP02")
    missing = [valid] * EXPECTED_COUNTY_ROWS
    missing[-1] = ""
    with pytest.raises(ValueError, match="missing GEO_ID"):
        validate_county_geoids(pd.DataFrame({"GEO_ID": missing}), "DP02")


def test_real_zip_pipeline_selects_exact_components_and_outputs_county_schema(tmp_path):
    _require_real_zips()
    interim, processed = tmp_path / "interim", tmp_path / "processed"
    manifest = preprocess_acs(RAW_ACS_DIR, interim, processed)
    expected_features = list(COMPONENTS)
    output = pd.read_csv(processed / "acs_features.csv", dtype={"county_fips": "string"})
    report = pd.read_csv(processed / "acs_validation_report.csv", dtype="string")

    assert list(output.columns) == ["county_fips", *expected_features]
    assert len(output) == EXPECTED_COUNTY_ROWS
    assert output["county_fips"].is_unique
    assert output["county_fips"].str.fullmatch(r"\d{5}").all()
    assert report["table"].tolist() == list(TABLES)
    assert report["status"].eq("valid").all()
    assert report["selected_features"].astype(int).sum() == len(expected_features)
    assert set(manifest) == set(TABLES)
