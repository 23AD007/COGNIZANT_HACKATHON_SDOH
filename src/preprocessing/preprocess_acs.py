"""Metadata-driven cleaning of selected ACS 2024 SDOH source variables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TABLES = ("DP02", "DP03", "DP04", "DP05", "S0101", "S1501", "S1701", "S1901")
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


def preprocess_acs(source_dir: str | Path, output_dir: str | Path) -> dict[str, dict[str, dict[str, str]]]:
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, dict[str, str]]] = {}
    for table in TABLES:
        data_path, metadata_path = table_paths(source_dir, table)
        cleaned, mapping = read_and_clean_table(data_path, metadata_path, table)
        cleaned.to_csv(output_dir / f"{table.lower()}_clean.csv", index=False)
        manifest[table] = mapping
    (output_dir / "metadata_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean selected ACS 2024 variables from metadata.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/acs"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/acs_clean"))
    args = parser.parse_args()
    manifest = preprocess_acs(args.source_dir, args.output_dir)
    print(f"Processed {len(manifest)} ACS tables into {args.output_dir}")


if __name__ == "__main__":
    main()
