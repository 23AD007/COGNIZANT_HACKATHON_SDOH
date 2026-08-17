from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member_model_features.csv"
)


@dataclass
class CanonicalMemberRecord:
    """
    Canonical representation of one HealthLens member.

    The record intentionally allows partial information.

    Missing source attributes remain missing/unknown.
    They are never converted into negative values automatically.
    """

    member_id: str

    clinical: dict[str, Any] = field(default_factory=dict)
    sdoh: dict[str, Any] = field(default_factory=dict)
    geography: dict[str, Any] = field(default_factory=dict)

    source: str | None = None
    source_record_id: str | None = None

    provenance: dict[str, Any] = field(default_factory=dict)

    def available_clinical_features(self) -> list[str]:
        return [
            key
            for key, value in self.clinical.items()
            if not _is_missing(value)
        ]

    def available_sdoh_features(self) -> list[str]:
        return [
            key
            for key, value in self.sdoh.items()
            if not _is_missing(value)
        ]

    def available_geography(self) -> list[str]:
        return [
            key
            for key, value in self.geography.items()
            if not _is_missing(value)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "clinical": dict(self.clinical),
            "sdoh": dict(self.sdoh),
            "geography": dict(self.geography),
            "source": self.source,
            "source_record_id": self.source_record_id,
            "provenance": dict(self.provenance),
        }


@dataclass
class CanonicalSchema:
    """
    Schema contract derived from the existing HealthLens
    member model feature file.

    This class does not create new model features.
    """

    member_id_column: str
    model_features: list[str]

    clinical_features: list[str] = field(default_factory=list)
    sdoh_features: list[str] = field(default_factory=list)
    geography_features: list[str] = field(default_factory=list)

    source_columns: list[str] = field(default_factory=list)

    def contains_model_feature(self, name: str) -> bool:
        return name in self.model_features

    def validate_feature_names(
        self,
        columns: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Returns:
            supported columns
            unsupported columns

        Unsupported columns are reported, not silently interpreted.
        """
        supported = [
            column
            for column in columns
            if column in self.model_features
        ]

        unsupported = [
            column
            for column in columns
            if column not in self.model_features
            and column != self.member_id_column
        ]

        return supported, unsupported

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id_column": self.member_id_column,
            "model_features": list(self.model_features),
            "clinical_features": list(self.clinical_features),
            "sdoh_features": list(self.sdoh_features),
            "geography_features": list(self.geography_features),
            "source_columns": list(self.source_columns),
        }


def load_existing_member_schema(
    feature_file: str | Path = DEFAULT_MODEL_FEATURE_FILE,
) -> CanonicalSchema:
    """
    Load the existing HealthLens member model schema.

    IMPORTANT:
    The existing CSV is the source of truth.
    No model feature names are invented here.
    """

    path = Path(feature_file)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Existing member model feature file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Existing member model feature file is empty:\n{path}"
        )

    columns = list(df.columns)

    member_id_column = _detect_member_id_column(columns)

    model_features = [
        column
        for column in columns
        if column != member_id_column
    ]

    if not model_features:
        raise ValueError(
            "No model features were found in "
            f"{path}"
        )

    return CanonicalSchema(
        member_id_column=member_id_column,
        model_features=model_features,
        source_columns=columns,
    )


def _detect_member_id_column(columns: list[str]) -> str:
    """
    Detect the existing member identifier column.

    Only accepts an actual column from the source schema.
    """

    candidates = [
        "member_id",
        "MemberID",
        "MEMBER_ID",
        "patient_id",
        "PATIENT_ID",
        "id",
        "ID",
    ]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        "Unable to identify member ID column. "
        f"Available columns: {columns}"
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

    except (TypeError, ValueError):
        pass

    return False


def save_schema(
    schema: CanonicalSchema,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            schema.to_dict(),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return path


if __name__ == "__main__":
    print("=" * 70)
    print("HEALTHLENS CANONICAL SCHEMA")
    print("=" * 70)

    schema = load_existing_member_schema()

    print(f"Schema source:       {DEFAULT_MODEL_FEATURE_FILE}")
    print(f"Member ID column:    {schema.member_id_column}")
    print(f"Model features:      {len(schema.model_features)}")

    print("\nExisting model features:")
    for index, feature in enumerate(schema.model_features):
        print(f"  {index:03d}. {feature}")

    print("\nCANONICAL SCHEMA SELF-TEST: PASSED")
    print("=" * 70)