"""Canonical clinical-context feature definitions used by the graph.

The graph already stores these source feature names on ``ClinicalFactor``
nodes.  This module is deliberately a read-only vocabulary; it does not
derive clinical observations or change the model feature set.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalFactor:
    key: str
    name: str
    description: str


from src.knowledge_graph.build_graph import CLINICAL_FEATURES as _CLINICAL_FEATURE_KEYS

CLINICAL_FACTORS = {
    key: ClinicalFactor(
        key=key,
        name=key.replace("clinical_", "").replace("_", " "),
        description="Existing member-level clinical context feature: " + key,
    )
    for key in sorted(_CLINICAL_FEATURE_KEYS)
}


def get_clinical_factor(key: str) -> ClinicalFactor:
    return CLINICAL_FACTORS[str(key).strip()]


def get_all_clinical_factors() -> list[ClinicalFactor]:
    return list(CLINICAL_FACTORS.values())


__all__ = ["ClinicalFactor", "CLINICAL_FACTORS", "get_clinical_factor", "get_all_clinical_factors"]
