"""Read-only relationships projected from the validated Knowledge Base."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .registry import KnowledgeBaseRegistry


@dataclass(frozen=True)
class KnowledgeBaseRelationship:
    source_id: str
    relationship_type: str
    target_id: str


def build_relationships(registry: KnowledgeBaseRegistry) -> list[KnowledgeBaseRelationship]:
    """Project the registry indexes without mutating graph or KB state."""
    relationships: list[KnowledgeBaseRelationship] = []
    for factor_id, factor in registry.factors.items():
        for domain in registry._references(factor, ("domain", "domain_key", "domain_id", "sdoh_domain")):
            resolved = registry.resolve_domain(domain)
            if resolved:
                relationships.append(KnowledgeBaseRelationship(factor_id, "BELONGS_TO_DOMAIN", resolved))
    for factor_id, intervention_ids in registry.interventions_by_factor.items():
        relationships.extend(KnowledgeBaseRelationship(factor_id, "SUPPORTED_BY_INTERVENTION", intervention_id)
                             for intervention_id in intervention_ids)
    for factor_id, evidence_ids in registry.evidence_by_factor.items():
        relationships.extend(KnowledgeBaseRelationship(factor_id, "SUPPORTED_BY_EVIDENCE", evidence_id)
                             for evidence_id in evidence_ids)
    return relationships


def relationships_by_source(relationships: Iterable[KnowledgeBaseRelationship], source_id: str) -> list[KnowledgeBaseRelationship]:
    return [relationship for relationship in relationships if relationship.source_id == source_id]


__all__ = ["KnowledgeBaseRelationship", "build_relationships", "relationships_by_source"]
