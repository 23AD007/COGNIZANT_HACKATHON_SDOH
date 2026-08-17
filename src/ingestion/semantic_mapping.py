"""Mapping of external columns to the existing member-risk feature schema."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Any, Mapping

from .canonical_schema import CanonicalSchema


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


@dataclass(frozen=True)
class FieldMapping:
    source_field: str
    canonical_feature: str | None
    method: str
    confidence: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticMapper:
    """Conservative mapper derived exclusively from an existing schema.

    Aliases are supplied by a caller/configuration rather than guessed from
    natural language. This keeps ambiguous health concepts out of inference
    and training until an owner reviews them.
    """

    def __init__(self, schema: CanonicalSchema, aliases: Mapping[str, str] | None = None) -> None:
        self.schema = schema
        self.aliases = dict(aliases or {})
        supported = [schema.member_id_column, *schema.model_features]
        self._normalised: dict[str, list[str]] = {}
        for feature in supported:
            self._normalised.setdefault(_normalise(feature), []).append(feature)

    def map_field(self, source_field: str) -> FieldMapping:
        if source_field in self._normalised.get(_normalise(source_field), []):
            return FieldMapping(source_field, source_field, "exact_match", 1.0, "EXACT")
        matches = self._normalised.get(_normalise(source_field), [])
        if len(matches) == 1:
            return FieldMapping(source_field, matches[0], "normalized_name", 0.98, "NORMALIZED")
        if source_field in self.aliases:
            target = self.aliases[source_field]
            if target in [self.schema.member_id_column, *self.schema.model_features]:
                return FieldMapping(source_field, target, "configured_alias", 0.90, "ALIAS")
        if len(matches) > 1:
            return FieldMapping(source_field, None, "ambiguous_normalized_name", 0.0, "REVIEW_REQUIRED")
        return FieldMapping(source_field, None, "none", 0.0, "UNMAPPED")

    def map_columns(self, columns: list[str]) -> list[FieldMapping]:
        return [self.map_field(column) for column in columns]
