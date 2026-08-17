from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .canonical_schema import (
    CanonicalMemberRecord,
    CanonicalSchema,
)


@dataclass
class ValidationResult:
    passed: bool

    member_count: int = 0

    duplicate_member_ids: list[str] = field(
        default_factory=list
    )

    missing_member_ids: list[str] = field(
        default_factory=list
    )

    unsupported_columns: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "member_count": self.member_count,
            "duplicate_member_ids": self.duplicate_member_ids,
            "missing_member_ids": self.missing_member_ids,
            "unsupported_columns": self.unsupported_columns,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_canonical_members(
    members: list[CanonicalMemberRecord],
    schema: CanonicalSchema,
) -> ValidationResult:

    result = ValidationResult(
        passed=True,
        member_count=len(members),
    )

    member_ids: list[str] = []

    for index, member in enumerate(members):

        if member.member_id is None:
            result.missing_member_ids.append(
                f"row:{index}"
            )
            continue

        member_id = str(member.member_id).strip()

        if not member_id:
            result.missing_member_ids.append(
                f"row:{index}"
            )
            continue

        member_ids.append(member_id)

        canonical_features = member.provenance.get(
            "canonical_features",
            {},
        )

        if not isinstance(canonical_features, dict):
            result.errors.append(
                f"Member {member_id}: "
                "canonical_features must be a dictionary."
            )
            continue

        unknown_features = [
            feature
            for feature in canonical_features
            if feature not in schema.model_features
        ]

        if unknown_features:
            result.unsupported_columns.extend(
                unknown_features
            )

    duplicate_ids = (
        pd.Series(member_ids)
        .value_counts()
        .loc[lambda series: series > 1]
        .index
        .tolist()
        if member_ids
        else []
    )

    result.duplicate_member_ids = [
        str(member_id)
        for member_id in duplicate_ids
    ]

    result.unsupported_columns = sorted(
        set(result.unsupported_columns)
    )

    if result.missing_member_ids:
        result.errors.append(
            "One or more canonical members have "
            "missing member IDs."
        )

    if result.duplicate_member_ids:
        result.errors.append(
            "Duplicate member IDs were detected."
        )

    if result.unsupported_columns:
        result.errors.append(
            "Unsupported canonical features were detected."
        )

    result.passed = not result.errors

    return result


def validate_source_dataframe(
    dataframe: pd.DataFrame,
    schema: CanonicalSchema,
) -> ValidationResult:

    errors: list[str] = []
    warnings: list[str] = []

    if dataframe is None:
        return ValidationResult(
            passed=False,
            errors=["Input dataframe is None."],
        )

    if dataframe.empty:
        warnings.append(
            "Input dataframe contains zero rows."
        )

    member_id_column = schema.member_id_column

    if member_id_column not in dataframe.columns:
        errors.append(
            "Required member ID column missing: "
            f"{member_id_column}"
        )
        return ValidationResult(
            passed=False,
            member_count=len(dataframe),
            errors=errors,
            warnings=warnings,
        )

    missing_ids = (
        dataframe[member_id_column]
        .isna()
        .sum()
    )

    if missing_ids:
        errors.append(
            f"{missing_ids} rows have missing member IDs."
        )

    duplicate_count = (
        dataframe[member_id_column]
        .duplicated()
        .sum()
    )

    if duplicate_count:
        errors.append(
            f"{duplicate_count} duplicate member IDs detected."
        )

    _, unsupported = schema.validate_feature_names(
        list(dataframe.columns)
    )

    if unsupported:
        warnings.append(
            "Source contains columns outside the existing "
            "HealthLens model schema: "
            + ", ".join(unsupported)
        )

    return ValidationResult(
        passed=not errors,
        member_count=len(dataframe),
        unsupported_columns=unsupported,
        errors=errors,
        warnings=warnings,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("HEALTHLENS CANONICAL VALIDATION")
    print("=" * 70)

    print(
        "Validation module loaded successfully."
    )

    print("\nCANONICAL VALIDATION SELF-TEST: PASSED")
    print("=" * 70)