"""Create a reproducible member-level extract from an extracted Synthea bundle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_REFERENCE_DATE = "2025-01-01"
MAX_VALID_AGE = 120
MEMBER_COLUMNS = [
    "member_id",
    "birthdate",
    "age",
    "gender",
    "race",
    "ethnicity",
    "marital_status",
    "city",
    "state",
    "zip",
]
REQUIRED_PATIENT_COLUMNS = {"id", "birthdate", "gender", "race", "ethnicity"}


def to_snake_case(column_name: str) -> str:
    """Return a deterministic snake_case representation of a source column."""
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", str(column_name).strip())
    return re.sub(r"_+", "_", normalized).strip("_").lower()


def find_patient_table(source_dir: str | Path) -> Path:
    """Find a Synthea patient/member CSV using filename and required columns."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Synthea source directory not found: {source_dir}")

    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {source_dir}")

    preferred = [path for path in csv_files if to_snake_case(path.stem) in {"patients", "patient", "members", "member"}]
    for path in [*preferred, *[item for item in csv_files if item not in preferred]]:
        columns = pd.read_csv(path, nrows=0).columns
        normalized_columns = {to_snake_case(column) for column in columns}
        if REQUIRED_PATIENT_COLUMNS.issubset(normalized_columns):
            return path

    raise ValueError(
        "Could not identify a Synthea patient table. "
        f"Expected columns: {sorted(REQUIRED_PATIENT_COLUMNS)}"
    )


def calculate_age(birthdate: pd.Series, reference_date: pd.Timestamp) -> pd.Series:
    """Calculate completed years of age as of a fixed reference date."""
    age = reference_date.year - birthdate.dt.year
    before_birthday = (birthdate.dt.month > reference_date.month) | (
        (birthdate.dt.month == reference_date.month) & (birthdate.dt.day > reference_date.day)
    )
    return (age - before_birthday.astype("Int64")).astype("Int64")


def clean_text(series: pd.Series) -> pd.Series:
    """Strip string fields while retaining missing values as pandas nullable strings."""
    return series.astype("string").str.strip().replace("", pd.NA)


def build_validation_report(
    cleaned: pd.DataFrame,
    raw_row_count: int,
    raw_column_count: int,
    invalid_date_count: int,
    invalid_age_count: int,
    duplicate_member_id_count: int,
    rows_dropped_missing_member_id: int,
    rows_dropped_duplicate_member_id: int,
    reference_date: pd.Timestamp,
) -> dict[str, Any]:
    """Build a JSON-serializable validation report for the cleaned member file."""
    geography_columns = ["city", "state", "zip"]
    return {
        "reference_date": reference_date.date().isoformat(),
        "source": "Synthea patient/member table",
        "raw_row_count": raw_row_count,
        "raw_column_count": raw_column_count,
        "cleaned_row_count": len(cleaned),
        "cleaned_column_count": len(cleaned.columns),
        "missing_values": {column: int(cleaned[column].isna().sum()) for column in cleaned.columns},
        "duplicate_member_ids_in_source": duplicate_member_id_count,
        "duplicate_member_ids_in_output": int(cleaned["member_id"].duplicated().sum()),
        "invalid_dates_in_source": invalid_date_count,
        "invalid_ages_in_source": invalid_age_count,
        "rows_dropped_missing_member_id": rows_dropped_missing_member_id,
        "rows_dropped_duplicate_member_id": rows_dropped_duplicate_member_id,
        "geographic_coverage": {
            column: {
                "non_missing_count": int(cleaned[column].notna().sum()),
                "missing_count": int(cleaned[column].isna().sum()),
                "coverage_pct": round(float(cleaned[column].notna().mean() * 100), 2),
            }
            for column in geography_columns
        },
    }


def preprocess_synthea(
    source_dir: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    reference_date: str | pd.Timestamp = DEFAULT_REFERENCE_DATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean Synthea patients and write a member file plus validation report."""
    patient_path = find_patient_table(source_dir)
    output_path = Path(output_path)
    report_path = Path(report_path)
    reference_timestamp = pd.Timestamp(reference_date).normalize()

    raw = pd.read_csv(patient_path, dtype="string", keep_default_na=True)
    raw.columns = [to_snake_case(column) for column in raw.columns]
    if raw.columns.duplicated().any():
        duplicates = raw.columns[raw.columns.duplicated()].tolist()
        raise ValueError(f"Column normalization produced duplicate names: {duplicates}")

    missing_required = REQUIRED_PATIENT_COLUMNS - set(raw.columns)
    if missing_required:
        raise ValueError(f"Patient table is missing required columns: {sorted(missing_required)}")

    raw_row_count, raw_column_count = raw.shape
    raw_birthdate = clean_text(raw["birthdate"])
    parsed_birthdate = pd.to_datetime(raw_birthdate, errors="coerce")
    invalid_date_count = int((raw_birthdate.notna() & parsed_birthdate.isna()).sum())

    cleaned = pd.DataFrame(index=raw.index)
    cleaned["member_id"] = clean_text(raw["id"])
    cleaned["birthdate"] = parsed_birthdate
    for source_column, target_column in [
        ("gender", "gender"),
        ("race", "race"),
        ("ethnicity", "ethnicity"),
        ("marital", "marital_status"),
        ("city", "city"),
        ("state", "state"),
        ("zip", "zip"),
    ]:
        cleaned[target_column] = clean_text(raw[source_column]) if source_column in raw else pd.Series(pd.NA, index=raw.index, dtype="string")

    cleaned["age"] = calculate_age(cleaned["birthdate"], reference_timestamp)
    invalid_age = cleaned["age"].isna() | (cleaned["age"] < 0) | (cleaned["age"] > MAX_VALID_AGE)
    invalid_age_count = int(invalid_age.sum())
    cleaned.loc[invalid_age, "age"] = pd.NA

    missing_member_id = cleaned["member_id"].isna()
    rows_dropped_missing_member_id = int(missing_member_id.sum())
    cleaned = cleaned.loc[~missing_member_id].copy()
    duplicate_member_id_count = int(cleaned["member_id"].duplicated().sum())
    cleaned = cleaned.drop_duplicates(subset="member_id", keep="first").copy()
    rows_dropped_duplicate_member_id = duplicate_member_id_count
    cleaned = cleaned[MEMBER_COLUMNS]

    if cleaned.empty:
        raise ValueError("No members remain after preprocessing.")
    if not cleaned["member_id"].is_unique:
        raise ValueError("Cleaned member IDs are not unique.")

    report = build_validation_report(
        cleaned=cleaned,
        raw_row_count=raw_row_count,
        raw_column_count=raw_column_count,
        invalid_date_count=invalid_date_count,
        invalid_age_count=invalid_age_count,
        duplicate_member_id_count=duplicate_member_id_count,
        rows_dropped_missing_member_id=rows_dropped_missing_member_id,
        rows_dropped_duplicate_member_id=rows_dropped_duplicate_member_id,
        reference_date=reference_timestamp,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return cleaned, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess an extracted Synthea patient file.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/synthea"))
    parser.add_argument("--output-path", type=Path, default=Path("data/interim/synthea_clean/members.csv"))
    parser.add_argument("--report-path", type=Path, default=Path("data/interim/synthea_clean/validation_report.json"))
    parser.add_argument("--reference-date", default=DEFAULT_REFERENCE_DATE)
    args = parser.parse_args()

    cleaned, report = preprocess_synthea(
        source_dir=args.source_dir,
        output_path=args.output_path,
        report_path=args.report_path,
        reference_date=args.reference_date,
    )
    print(f"Detected patient table: {find_patient_table(args.source_dir)}")
    print(f"Saved {len(cleaned)} members to {args.output_path}")
    print(f"Saved validation report to {args.report_path}")
    print(json.dumps({key: report[key] for key in ("cleaned_row_count", "invalid_dates_in_source", "invalid_ages_in_source")}, indent=2))


if __name__ == "__main__":
    main()
