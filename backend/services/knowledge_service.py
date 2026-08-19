"""Read-only views of the persisted HealthLens knowledge graph."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from src.knowledge_graph.queries import KnowledgeGraphQuery


EXPOSED_NODE_TYPES = {"SDOH_DOMAIN", "SDOH_FACTOR", "Evidence", "INTERVENTION"}


class KnowledgeGraphUnavailable(RuntimeError):
    """The persisted graph cannot be loaded for the demo API."""


@lru_cache(maxsize=1)
def _graph_query() -> KnowledgeGraphQuery:
    try:
        return KnowledgeGraphQuery()
    except FileNotFoundError as exc:
        raise KnowledgeGraphUnavailable("Knowledge graph artifact was not found.") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise KnowledgeGraphUnavailable("Knowledge graph artifact is unavailable or invalid.") from exc


def _node_record(node: dict[str, Any]) -> dict[str, Any] | None:
    node_id = node.get("node_id")
    node_type = node.get("node_type")
    if not isinstance(node_id, str) or not isinstance(node_type, str):
        return None
    record: dict[str, Any] = {"node_id": node_id, "node_type": node_type}
    if isinstance(node.get("label"), str):
        record["label"] = node["label"]
    if isinstance(node.get("properties"), dict):
        record["properties"] = node["properties"]
    return record


def _relationship_record(edge: dict[str, Any]) -> dict[str, Any] | None:
    source = edge.get("source")
    target = edge.get("target")
    relationship_type = edge.get("relationship_type")
    if not all(isinstance(value, str) for value in (source, target, relationship_type)):
        return None
    record: dict[str, Any] = {"source": source, "target": target, "relationship_type": relationship_type}
    if isinstance(edge.get("edge_id"), str):
        record["edge_id"] = edge["edge_id"]
    if isinstance(edge.get("properties"), dict):
        record["properties"] = edge["properties"]
    return record


def get_knowledge_graph() -> dict[str, Any]:
    """Return actual non-member knowledge nodes and their graph relationships."""
    query = _graph_query()
    statistics = query.graph_statistics()
    nodes = [record for node in query.nodes if node.get("node_type") in EXPOSED_NODE_TYPES if (record := _node_record(node))]
    exposed_ids = {node["node_id"] for node in nodes}
    relationships = [record for edge in query.edges if edge.get("source") in exposed_ids and edge.get("target") in exposed_ids if (record := _relationship_record(edge))]
    return {
        "schema_version": statistics["schema_version"],
        "node_count": statistics["nodes"],
        "relationship_count": statistics["relationships"],
        "node_types": statistics["node_types"],
        "relationship_types": statistics["relationship_types"],
        "nodes": nodes,
        "relationships": relationships,
    }
