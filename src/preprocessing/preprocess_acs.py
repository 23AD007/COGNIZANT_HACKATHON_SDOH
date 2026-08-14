"""Metadata-driven cleaning of selected ACS 2024 SDOH source variables."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import BinaryIO

import pandas as pd


TABLES = ("DP02", "DP03", "DP04", "DP05", "S0101", "S1501", "S1701", "S1901")
COUNTY_GEOID_PATTERN = re.compile(r"^0500000US\d{5}$")
EXPECTED_COUNTY_ROWS = 3222
ZIP_FILENAMES = {
    "DP02": "ACSDP5Y2024.DP02_2026-08-14T102208.zip",
    "DP03": "ACSDP5Y2024.DP03_2026-08-14T100149.zip",
    "DP04": "ACSDP5Y2024.DP04_2026-08-14T103221.zip",
    "DP05": "ACSDP5Y2024.DP05_2026-08-14T103538.zip",
    "S0101": "ACSST5Y2024.S0101_2026-08-14T104130.zip",
    "S1501": "ACSST5Y2024.S1501_2026-08-14T104417.zip",
    "S1701": "ACSST5Y2024.S1701_2026-08-14T104646.zip",
    "S1901": "ACSST5Y2024.S1901_2026-08-14T105139.zip",
}
# Only documented source variables used by src.features.acs_features are retained.
COMPONENTS = {
    "education_less_than_9th_pct": ("DP02", "DP02_0060PE"),
    "education_9th_to_12th_no_diploma_pct": ("DP02", "DP02_0061PE"),
    "digital_with_computer_pct": ("DP02", "DP02_0153PE"),
    "digital_with_broadband_pct": ("DP02", "DP02_0154PE"),
    "unemployment_pct": ("DP03", "DP03_0005PE"),
    "public_assistance_pct": ("DP03", "DP03_0072PE"),
    "uninsured_pct": ("DP03", "DP03_0099PE"),
    "mean_commute_minutes": ("DP03", "DP03_0025E"),
    "housing_vacancy_pct": ("DP04", "DP04_0003PE"),
    "housing_no_vehicle_pct": ("DP04", "DP04_0058PE"),
    "housing_renter_pct": ("DP04", "DP04_0047PE"),
    "housing_crowded_1_01_to_1_50_pct": ("DP04", "DP04_0078PE"),
    "housing_crowded_1_51_plus_pct": ("DP04", "DP04_0079PE"),
    "housing_rent_30_to_34_9_pct": ("DP04", "DP04_0141PE"),
    "housing_rent_35_plus_pct": ("DP04", "DP04_0142PE"),
    "population_total": ("S0101", "S0101_C01_001E"),
    "median_age_years": ("S0101", "S0101_C01_032E"),
    "female_pct": ("S0101", "S0101_C06_001E"),
    "education_high_school_pct": ("S1501", "S1501_C02_009E"),
    "education_college_pct": ("S1501", "S1501_C02_010E"),
    "education_bachelors_or_higher_pct": ("S1501", "S1501_C02_015E"),
    "poverty_pct": ("S1701", "S1701_C03_001E"),
    "median_household_income": ("S1901", "S1901_C01_012E"),
}
SUPPRESSION_MARKERS = {"", "-", "--", "***", "**", "(X)", "N", "NA", "N/A", "NULL", "null"}


def geographic_level(geo_id: str) -> str:
    """Classify Census GEO_ID prefixes without changing their original values."""
    prefixes = {"010": "national", "020": "region", "030": "division", "040": "state", "050": "county", "140": "census_tract"}
    return prefixes.get(str(geo_id)[:3], "unknown")


def clean_numeric(values: pd.Series) -> pd.Series:
    """Convert ACS numeric strings while mapping suppression markers to missing."""
    text = values.astype("string").str.strip().replace(list(SUPPRESSION_MARKERS), pd.NA)
    return pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")


def read_metadata(path: Path) -> dict[str, str]:
    metadata = pd.read_csv(path, dtype="string")
    required = {"Column Name", "Label"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata file lacks {sorted(required)}: {path}")
    return dict(zip(metadata["Column Name"], metadata["Label"], strict=True))


def selected_components(table: str) -> dict[str, str]:
    """Return only the verified components for one ACS table."""
    return {alias: code for alias, (component_table, code) in COMPONENTS.items() if component_table == table}


def discover_zip_inputs(source_dir: str | Path) -> dict[str, Path]:
    """Locate only the eight exact, verified ACS ZIP filenames."""
    source_dir = Path(source_dir)
    discovered: dict[str, Path] = {}
    for table in TABLES:
        source_zip = source_dir / ZIP_FILENAMES[table]
        if not source_zip.is_file():
            raise FileNotFoundError(f"Required ACS ZIP is missing: {source_zip}")
        discovered[table] = source_zip
    return discovered


def _single_zip_member(archive: zipfile.ZipFile, suffix: str, source_zip: Path) -> zipfile.ZipInfo:
    matches = [info for info in archive.infolist() if not info.is_dir() and info.filename.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"{source_zip} must contain exactly one {suffix}; found {len(matches)}")
    return matches[0]


def zip_members(source_zip: str | Path, table: str) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo]:
    """Validate a ZIP and locate its table-specific Data and Metadata CSVs."""
    source_zip = Path(source_zip)
    try:
        with zipfile.ZipFile(source_zip) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"ZIP integrity check failed for {source_zip}")
            return (
                _single_zip_member(archive, f"{table}-Data.csv", source_zip),
                _single_zip_member(archive, f"{table}-Column-Metadata.csv", source_zip),
            )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid ZIP archive: {source_zip}") from exc


def _read_metadata(source: Path | BinaryIO) -> dict[str, str]:
    metadata = pd.read_csv(source, dtype="string")
    required = {"Column Name", "Label"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata lacks {sorted(required)}")
    return dict(zip(metadata["Column Name"], metadata["Label"], strict=True))


def validate_county_geoids(frame: pd.DataFrame, table: str) -> None:
    """Require unique, complete county GEO_IDs in the verified format and count."""
    if "GEO_ID" not in frame.columns:
        raise ValueError(f"{table} lacks GEO_ID")
    geoids = frame["GEO_ID"].astype("string").str.strip()
    if geoids.isna().any() or geoids.eq("").any():
        raise ValueError(f"{table} has missing GEO_ID")
    if not geoids.str.fullmatch(COUNTY_GEOID_PATTERN.pattern, na=False).all():
        raise ValueError(f"{table} has invalid county GEO_ID")
    if geoids.duplicated().any():
        raise ValueError(f"{table} has duplicate GEO_ID")
    if len(frame) != EXPECTED_COUNTY_ROWS:
        raise ValueError(f"{table} must contain exactly {EXPECTED_COUNTY_ROWS} county records; found {len(frame)}")


def read_zip_table(source_zip: str | Path, table: str) -> tuple[pd.DataFrame, dict[str, str], int]:
    """Read Data and Metadata CSVs directly from a ZIP without extracting them."""
    source_zip = Path(source_zip)
    zip_members(source_zip, table)
    with zipfile.ZipFile(source_zip) as archive:
        data_member = _single_zip_member(archive, f"{table}-Data.csv", source_zip)
        metadata_member = _single_zip_member(archive, f"{table}-Column-Metadata.csv", source_zip)
        with archive.open(data_member) as data_file:
            raw = pd.read_csv(data_file, dtype="string", keep_default_na=False)
        with archive.open(metadata_member) as metadata_file:
            labels = _read_metadata(metadata_file)
    return raw, labels, len(raw)


def clean_county_table(raw: pd.DataFrame, labels: dict[str, str], table: str) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Remove only the Census export-label row and select verified county features."""
    if not {"GEO_ID", "NAME"}.issubset(raw.columns):
        raise ValueError(f"{table} lacks GEO_ID/NAME")
    raw = raw.loc[raw["GEO_ID"].astype("string").str.strip().ne("Geography")].copy()
    raw["GEO_ID"] = raw["GEO_ID"].astype("string").str.strip()
    raw["NAME"] = raw["NAME"].astype("string").str.strip()
    validate_county_geoids(raw, table)

    selected = selected_components(table)
    missing = sorted(set(selected.values()) - set(raw.columns))
    if missing:
        raise ValueError(f"{table} missing selected source columns: {missing}")

    cleaned = raw[["GEO_ID", "NAME"]].copy()
    cleaned["county_fips"] = cleaned["GEO_ID"].str.removeprefix("0500000US")
    mapping: dict[str, dict[str, str]] = {}
    for alias, code in selected.items():
        label = labels.get(code)
        if not label:
            raise ValueError(f"{table} metadata has no label for {code}")
        cleaned[alias] = clean_numeric(raw[code])
        mapping[alias] = {"source_code": code, "metadata_label": label}
    return cleaned, mapping


def table_paths(source_dir: Path, table: str) -> tuple[Path, Path]:
    data = list(source_dir.glob(f"*{table}-Data.csv"))
    metadata = list(source_dir.glob(f"*{table}-Column-Metadata.csv"))
    if len(data) != 1 or len(metadata) != 1:
        raise FileNotFoundError(f"Expected one data and metadata file for {table} in {source_dir}")
    return data[0], metadata[0]


def read_and_clean_table(data_path: Path, metadata_path: Path, table: str) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Read one ACS table and retain only documented, non-MOE feature components."""
    raw = pd.read_csv(data_path, dtype="string", keep_default_na=False)
    if not {"GEO_ID", "NAME"}.issubset(raw.columns):
        raise ValueError(f"{data_path} lacks GEO_ID/NAME")
    raw = raw.loc[raw["GEO_ID"].str.strip().ne("Geography")].copy()
    raw["GEO_ID"] = raw["GEO_ID"].astype("string").str.strip()
    raw["NAME"] = raw["NAME"].astype("string").str.strip()
    if raw["GEO_ID"].isna().any() or raw["GEO_ID"].eq("").any():
        raise ValueError(f"{table} has missing geographic keys")
    if raw["GEO_ID"].duplicated().any():
        raise ValueError(f"{table} has duplicate GEO_ID values")

    labels = read_metadata(metadata_path)
    selected = {alias: code for alias, (component_table, code) in COMPONENTS.items() if component_table == table}
    missing = set(selected.values()) - set(raw.columns)
    if missing:
        raise ValueError(f"{table} missing selected source columns: {sorted(missing)}")

    cleaned = raw[["GEO_ID", "NAME"]].copy()
    cleaned["geographic_level"] = cleaned["GEO_ID"].map(geographic_level)
    mapping: dict[str, dict[str, str]] = {}
    for alias, code in selected.items():
        label = labels.get(code)
        if not label:
            raise ValueError(f"{table} metadata has no label for {code}")
        if "margin of error" in label.lower() or code.endswith("M") or code.endswith("PM"):
            raise ValueError(f"Margin-of-error field cannot be a feature: {code}")
        if "race" in label.lower() or "hispanic" in label.lower() or "ethnicity" in label.lower():
            raise ValueError(f"Race/ethnicity field cannot be a risk feature: {code}")
        cleaned[alias] = clean_numeric(raw[code])
        mapping[alias] = {"source_code": code, "metadata_label": label}
    return cleaned, mapping


def preprocess_acs(
    source_dir: str | Path = "data/raw/acs",
    output_dir: str | Path = "data/interim/acs_clean",
    processed_dir: str | Path = "data/processed",
) -> dict[str, dict[str, dict[str, str]]]:
    """Create county ACS tables directly from the eight supplied ZIP archives."""
    source_dir, output_dir, processed_dir = Path(source_dir), Path(output_dir), Path(processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    zip_paths = discover_zip_inputs(source_dir)
    manifest: dict[str, dict[str, dict[str, str]]] = {}
    validation_rows: list[dict[str, object]] = []
    county_features: pd.DataFrame | None = None

    for table in TABLES:
        raw, labels, raw_rows = read_zip_table(zip_paths[table], table)
        cleaned, mapping = clean_county_table(raw, labels, table)
        cleaned.to_csv(output_dir / f"{table.lower()}_clean.csv", index=False)
        manifest[table] = mapping
        feature_columns = list(mapping)
        feature_frame = cleaned[["county_fips", *feature_columns]]
        county_features = (
            feature_frame
            if county_features is None
            else county_features.merge(feature_frame, on="county_fips", how="outer", validate="one_to_one")
        )
        validation_rows.append({
            "table": table,
            "source_zip": zip_paths[table].name,
            "raw_rows": raw_rows,
            "data_rows": len(cleaned),
            "selected_features": len(feature_columns),
            "missing_features": "",
            "invalid_geoids": 0,
            "duplicate_geoids": 0,
            "missing_geoids": 0,
            "county_count": cleaned["county_fips"].nunique(),
            "status": "valid",
        })

    assert county_features is not None
    if len(county_features) != EXPECTED_COUNTY_ROWS or county_features["county_fips"].isna().any():
        raise ValueError("ACS county feature merge did not preserve the validated county key set")
    county_features.to_csv(processed_dir / "acs_features.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(processed_dir / "acs_validation_report.csv", index=False)
    (output_dir / "metadata_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean selected ACS 2024 variables from metadata.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/acs"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/acs_clean"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    manifest = preprocess_acs(args.source_dir, args.output_dir, args.processed_dir)
    print(f"Processed {len(manifest)} county ACS tables from ZIP inputs")


if __name__ == "__main__":
    main()
