"""Normalize documented USDA SRAM tract components without changing raw sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FILES = {
    "general": ("SRAM General Tract Characteristics Data.csv", "SRAM Variable Lookup General Tract Characteristics.csv"),
    "driving": ("SRAM Driving Distance Data.csv", "SRAM Variable Lookup Driving Distance.csv"),
    "straight": ("SRAM Straight Line Distance Data.csv", "SRAM Variable Lookup Straight Line Distance.csv"),
}
SELECTED_FIELDS = {
    "general": ["CensusTract20", "State", "County20", "Urban", "POP2020", "OHU2020", "LowIncomeTracts", "PovertyRate", "TractHUNV", "TractSNAP"],
    "driving": ["CensusTract20", "DD_SRAM_LILATracts_1And10", "DD_SRAM_HUNVFlag", "DD_SRAM_LAPOP1_10", "DD_SRAM_lapop1share", "DD_SRAM_lahunv1", "DD_SRAM_lasnap1"],
    "straight": ["CensusTract20", "SD_SRAM_LILATracts_1And10", "SD_SRAM_HUNVFlag", "SD_SRAM_LAPOP1_10", "SD_SRAM_lapop1share", "SD_SRAM_lahunv1", "SD_SRAM_lasnap1"],
}
SUPPRESSION_MARKERS = {"", "-", "--", "**", "***", "(X)", "N", "NA", "N/A", "NULL", "null"}


def normalize_tract_geoid(values: pd.Series) -> pd.Series:
    """Normalize documented CensusTract20 values to 11-digit Census tract GEOIDs."""
    key = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    valid = key.str.fullmatch(r"\d{10,11}", na=False)
    if not valid.all():
        raise ValueError("CensusTract20 contains an invalid tract identifier.")
    return key.str.zfill(11)


def clean_numeric(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().replace(list(SUPPRESSION_MARKERS), pd.NA)
    return pd.to_numeric(text, errors="coerce")


def read_documented_table(source_dir: Path, table_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    data_filename, lookup_filename = FILES[table_name]
    data_path, lookup_path = source_dir / data_filename, source_dir / lookup_filename
    fields = SELECTED_FIELDS[table_name]
    header = pd.read_csv(data_path, nrows=0, encoding="latin-1")
    missing = set(fields) - set(header.columns)
    if missing:
        raise ValueError(f"{data_filename} missing fields: {sorted(missing)}")
    lookup = pd.read_csv(lookup_path, dtype="string", encoding="utf-8-sig")
    if not {"Field", "LongName", "Description"}.issubset(lookup.columns):
        raise ValueError(f"Invalid SRAM variable lookup: {lookup_filename}")
    metadata = {row.Field: f"{row.LongName} | {row.Description}" for row in lookup.itertuples(index=False)}
    undocumented = set(fields) - {"CensusTract20", "State", "County20"} - set(metadata)
    if undocumented:
        raise ValueError(f"Selected SRAM fields lack documentation: {sorted(undocumented)}")
    frame = pd.read_csv(data_path, usecols=fields, dtype="string", encoding="latin-1", low_memory=False)
    frame["tract_geoid"] = normalize_tract_geoid(frame["CensusTract20"])
    if frame["tract_geoid"].duplicated().any():
        raise ValueError(f"Duplicate tract GEOIDs in {data_filename}")
    for column in fields:
        if column not in {"CensusTract20", "State", "County20"}:
            frame[column] = clean_numeric(frame[column])
    return frame, {field: metadata[field] for field in fields if field in metadata}


def preprocess_sram(source_dir: str | Path, output_path: str | Path, manifest_path: str | Path) -> pd.DataFrame:
    source_dir = Path(source_dir)
    general, general_metadata = read_documented_table(source_dir, "general")
    driving, driving_metadata = read_documented_table(source_dir, "driving")
    straight, straight_metadata = read_documented_table(source_dir, "straight")
    components = general.merge(driving.drop(columns="CensusTract20"), on="tract_geoid", how="inner", validate="one_to_one")
    components = components.merge(straight.drop(columns="CensusTract20"), on="tract_geoid", how="inner", validate="one_to_one")
    components["state_fips"] = components["tract_geoid"].str[:2]
    components["county_fips"] = components["tract_geoid"].str[:5]
    components["tract_code"] = components["tract_geoid"].str[5:]
    output_path, manifest_path = Path(output_path), Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(output_path, index=False)
    manifest_path.write_text(json.dumps({"general": general_metadata, "driving": driving_metadata, "straight": straight_metadata}, indent=2) + "\n", encoding="utf-8")
    return components


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize documented USDA SRAM tract components.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/sram"))
    parser.add_argument("--output-path", type=Path, default=Path("data/interim/sram_clean/sram_tract_components.csv"))
    parser.add_argument("--manifest-path", type=Path, default=Path("data/interim/sram_clean/metadata_manifest.json"))
    args = parser.parse_args()
    result = preprocess_sram(args.source_dir, args.output_path, args.manifest_path)
    print(f"Saved {len(result)} normalized SRAM tract records to {args.output_path}")


if __name__ == "__main__":
    main()
