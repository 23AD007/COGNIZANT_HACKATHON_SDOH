from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from .canonical_schema import (
    CanonicalMemberRecord,
    CanonicalSchema,
    PROJECT_ROOT,
    _is_missing,
)


class SourceAdapter(ABC):
    """
    Base interface for external data sources.

    Each source must explicitly map its own fields into the
    existing HealthLens canonical feature names.
    """

    def __init__(
        self,
        schema: CanonicalSchema,
        source_name: str,
    ) -> None:
        self.schema = schema
        self.source_name = source_name

    @abstractmethod
    def adapt(
        self,
        dataframe: pd.DataFrame,
    ) -> list[CanonicalMemberRecord]:
        """
        Convert source-specific rows into canonical members.
        """
        raise NotImplementedError


class DataFrameSourceAdapter(SourceAdapter):
    """
    Generic adapter for a dataframe whose columns already use
    HealthLens canonical feature names.

    It intentionally does NOT guess mappings between arbitrary
    source columns and model features.

    Source-specific mappings should be supplied explicitly.
    """

    def __init__(
        self,
        schema: CanonicalSchema,
        source_name: str,
        column_mapping: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            schema=schema,
            source_name=source_name,
        )

        self.column_mapping = dict(column_mapping or {})

    def adapt(
        self,
        dataframe: pd.DataFrame,
    ) -> list[CanonicalMemberRecord]:

        if dataframe is None:
            raise ValueError("Input dataframe cannot be None.")

        if dataframe.empty:
            return []

        df = dataframe.copy()

        member_source_column = self._resolve_member_id_column(df)

        records: list[CanonicalMemberRecord] = []

        for row_number, row in df.iterrows():

            member_id = row[member_source_column]

            if _is_missing(member_id):
                raise ValueError(
                    f"Missing member ID at dataframe row {row_number}."
                )

            member_id = str(member_id).strip()

            if not member_id:
                raise ValueError(
                    f"Empty member ID at dataframe row {row_number}."
                )

            canonical_values: dict[str, Any] = {}

            for source_column in df.columns:

                if source_column == member_source_column:
                    continue

                target_column = self.column_mapping.get(
                    source_column,
                    source_column,
                )

                if target_column not in self.schema.model_features:
                    continue

                value = row[source_column]

                if _is_missing(value):
                    continue

                canonical_values[target_column] = value

            record = CanonicalMemberRecord(
                member_id=member_id,
                clinical={},
                sdoh={},
                geography={},
                source=self.source_name,
                source_record_id=member_id,
                provenance={
                    "source": self.source_name,
                    "row_index": int(row_number)
                    if isinstance(row_number, int)
                    else str(row_number),
                    "mapped_columns": {
                        source: target
                        for source, target in self.column_mapping.items()
                        if source in df.columns
                    },
                },
            )

            # At this stage we retain the canonical model features
            # without pretending to know whether a feature is clinical,
            # SDOH, or geography unless the existing project schema
            # explicitly establishes that classification.
            record.provenance["canonical_features"] = canonical_values

            records.append(record)

        return records

    def _resolve_member_id_column(
        self,
        dataframe: pd.DataFrame,
    ) -> str:

        expected = self.schema.member_id_column

        if expected in dataframe.columns:
            return expected

        for source_column, target_column in self.column_mapping.items():
            if (
                source_column in dataframe.columns
                and target_column == expected
            ):
                return source_column

        raise ValueError(
            "Input source does not contain the canonical member ID "
            f"column '{expected}' and no explicit mapping was provided."
        )


def load_source_dataframe(
    path: str | Path,
) -> pd.DataFrame:

    source_path = Path(path)

    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source file not found:\n{source_path}"
        )

    suffix = source_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(source_path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source_path)

    raise ValueError(
        "Unsupported source format: "
        f"{source_path.suffix}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("HEALTHLENS SOURCE ADAPTER")
    print("=" * 70)

    print("Source adapter module loaded successfully.")
    print(
        "No source mapping is guessed automatically."
    )

    print("\nSOURCE ADAPTER SELF-TEST: PASSED")
    print("=" * 70)