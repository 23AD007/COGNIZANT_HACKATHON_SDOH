"""CLI for the county-level SRAM aggregation; raw sources are never modified."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.sram_county_features import aggregate_sram_by_county


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate processed SRAM tract features by county FIPS.")
    parser.add_argument("--input-path", type=Path, default=Path("data/processed/sram_features.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/sram_county_features.csv"))
    parser.add_argument("--report-path", type=Path, default=Path("data/processed/sram_county_coverage_report.csv"))
    args = parser.parse_args()

    features = pd.read_csv(args.input_path, dtype={"tract_geoid": "string", "county_fips": "string", "state_fips": "string"})
    county, report = aggregate_sram_by_county(features)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    county.to_csv(args.output_path, index=False)
    pd.DataFrame([report]).to_csv(args.report_path, index=False)
    print(f"Saved {len(county)} county SRAM rows to {args.output_path}")


if __name__ == "__main__":
    main()
