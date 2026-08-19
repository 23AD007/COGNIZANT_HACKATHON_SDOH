from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeNode(BaseModel):
    node_id: str
    node_type: str
    label: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class KnowledgeRelationship(BaseModel):
    edge_id: str | None = None
    source: str
    target: str
    relationship_type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphResponse(BaseModel):
    schema_version: str | None = None
    node_count: int
    relationship_count: int
    node_types: dict[str, int]
    relationship_types: dict[str, int]
    nodes: list[KnowledgeNode]
    relationships: list[KnowledgeRelationship]
