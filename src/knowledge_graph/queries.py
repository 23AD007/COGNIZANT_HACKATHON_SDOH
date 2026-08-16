"""
HealthLens Knowledge Graph Query Layer
======================================

Reads the serialized HealthLens knowledge graph and provides
read-only query functions for:

    - member lookup
    - risk assessment
    - SDOH factors
    - SDOH domains
    - clinical factors
    - county context
    - interventions
    - evidence
    - aggregated member context
    - similar-member comparison
    - intervention reasoning

This module DOES NOT train or modify the ML model.

The ML model remains responsible for risk prediction.
The knowledge graph provides context, relationships, evidence,
and intervention reasoning.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"

GRAPH_DIR = PROCESSED_DIR / "knowledge_graph"

GRAPH_FILE = GRAPH_DIR / "healthlens_knowledge_graph.json"


# ============================================================================
# CONSTANTS
# ============================================================================

SCHEMA_VERSION = "1.0.0"

MEMBER_NODE_TYPE = "MEMBER"
COUNTY_NODE_TYPE = "COUNTY"
RISK_NODE_TYPE = "RiskAssessment"
SDOH_FACTOR_NODE_TYPE = "SDOH_FACTOR"
SDOH_DOMAIN_NODE_TYPE = "SDOH_DOMAIN"
CLINICAL_FACTOR_NODE_TYPE = "ClinicalFactor"
INTERVENTION_NODE_TYPE = "INTERVENTION"
EVIDENCE_NODE_TYPE = "Evidence"
COUNTY_RISK_NODE_TYPE = "CountyRiskAssessment"


# ============================================================================
# RELATIONSHIP TYPES
# ============================================================================

REL_HAS_RISK = "HAS_RISK_ASSESSMENT"
REL_HAS_SDOH = "HAS_SDOH_FACTOR"
REL_HAS_CLINICAL = "HAS_CLINICAL_CONTEXT"
REL_LIVES_IN = "LIVES_IN"
REL_RECEIVES_INTERVENTION = "RECEIVES_INTERVENTION_RECOMMENDATION"
REL_BELONGS_TO_DOMAIN = "BELONGS_TO_DOMAIN"
REL_ADDRESSES_DOMAIN = "ADDRESSES_DOMAIN"
REL_SUPPORTED_BY = "SUPPORTED_BY"


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def _safe_float(value: Any) -> float | None:
    """Convert a value to float without raising."""
    if value is None:
        return None

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    """Convert a value to int without raising."""
    if value is None:
        return None

    try:
        return int(value)

    except (TypeError, ValueError):
        return None


def _clean_string(value: Any) -> str | None:
    """Normalize strings and return None for empty values."""

    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

        return value

    return str(value)


def _first_non_null(
    data: dict[str, Any],
    keys: Iterable[str],
) -> Any:
    """Return the first non-null value for a list of possible keys."""

    for key in keys:

        if key in data:

            value = data[key]

            if value is not None:
                return value

    return None


def _normalise_properties(node: dict[str, Any]) -> dict[str, Any]:
    """
    Return node properties.

    The graph builder may serialize properties either directly on the
    node or inside a 'properties' dictionary.

    This function supports both forms.
    """

    if not isinstance(node, dict):
        return {}

    properties = node.get("properties")

    if isinstance(properties, dict):

        result = dict(properties)

        # Preserve useful top-level fields if they are not duplicated
        # inside properties.
        for key, value in node.items():

            if key not in {
                "properties",
                "type",
                "node_type",
                "id",
            }:
                result.setdefault(key, value)

        return result

    return {
        key: value
        for key, value in node.items()
        if key not in {
            "type",
            "node_type",
            "id",
        }
    }


def _node_type(node: dict[str, Any]) -> str | None:
    """Return normalized node type."""

    if not isinstance(node, dict):
        return None

    value = _first_non_null(
        node,
        (
            "type",
            "node_type",
            "label",
        ),
    )

    return _clean_string(value)


def _node_id(node: dict[str, Any]) -> str | None:
    """Return normalized node identifier."""

    if not isinstance(node, dict):
        return None

    value = _first_non_null(
        node,
        (
            "id",
            "node_id",
            "key",
        ),
    )

    return _clean_string(value)


def _edge_type(edge: dict[str, Any]) -> str | None:
    """Return normalized relationship type."""

    if not isinstance(edge, dict):
        return None

    value = _first_non_null(
        edge,
        (
            "type",
            "relationship_type",
            "label",
        ),
    )

    return _clean_string(value)


def _edge_source(edge: dict[str, Any]) -> str | None:
    """Return source node id from an edge."""

    if not isinstance(edge, dict):
        return None

    value = _first_non_null(
        edge,
        (
            "source",
            "source_id",
            "from",
            "from_id",
        ),
    )

    if isinstance(value, dict):
        return _node_id(value)

    return _clean_string(value)


def _edge_target(edge: dict[str, Any]) -> str | None:
    """Return target node id from an edge."""

    if not isinstance(edge, dict):
        return None

    value = _first_non_null(
        edge,
        (
            "target",
            "target_id",
            "to",
            "to_id",
        ),
    )

    if isinstance(value, dict):
        return _node_id(value)

    return _clean_string(value)


# ============================================================================
# GRAPH LOADING
# ============================================================================


class KnowledgeGraphQuery:
    """
    Read-only query interface over the serialized HealthLens graph.
    """

    def __init__(self, graph_file: Path | str = GRAPH_FILE):

        self.graph_file = Path(graph_file)

        if not self.graph_file.exists():
            raise FileNotFoundError(
                f"Knowledge graph not found:\n{self.graph_file}"
            )

        self.graph = self._load_graph()

        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []

        self.nodes_by_id: dict[str, dict[str, Any]] = {}

        self.outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)

        self._prepare_graph()

    # ------------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------------

    def _load_graph(self) -> dict[str, Any]:
        with self.graph_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            graph = json.load(file)

        if not isinstance(graph, dict):
            raise ValueError(
                "Knowledge graph JSON must contain an object."
            )

        return graph

    # ------------------------------------------------------------------------
    # PREPARE
    # ------------------------------------------------------------------------

    def _prepare_graph(self) -> None:

        self.nodes = self._extract_nodes()
        self.edges = self._extract_edges()

        for node in self.nodes:

            node_id = _node_id(node)

            if node_id is None:
                continue

            self.nodes_by_id[node_id] = node

        for edge in self.edges:

            source = _edge_source(edge)
            target = _edge_target(edge)

            if source is not None:
                self.outgoing[source].append(edge)

            if target is not None:
                self.incoming[target].append(edge)

    # ------------------------------------------------------------------------
    # NODE EXTRACTION
    # ------------------------------------------------------------------------

    def _extract_nodes(self) -> list[dict[str, Any]]:

        candidates = (
            self.graph.get("nodes"),
            self.graph.get("vertices"),
        )

        for candidate in candidates:

            if isinstance(candidate, list):
                return [
                    item
                    for item in candidate
                    if isinstance(item, dict)
                ]

        # Some graph serializers use a dictionary keyed by node id.
        for key in ("nodes", "vertices"):

            candidate = self.graph.get(key)

            if isinstance(candidate, dict):

                result = []

                for node_id, value in candidate.items():

                    if isinstance(value, dict):

                        node = dict(value)
                        node.setdefault("id", node_id)

                        result.append(node)

                return result

        return []

    # ------------------------------------------------------------------------
    # EDGE EXTRACTION
    # ------------------------------------------------------------------------

    def _extract_edges(self) -> list[dict[str, Any]]:

        candidates = (
            self.graph.get("relationships"),
            self.graph.get("edges"),
            self.graph.get("links"),
        )

        for candidate in candidates:

            if isinstance(candidate, list):
                return [
                    item
                    for item in candidate
                    if isinstance(item, dict)
                ]

        return []

    # =========================================================================
    # GRAPH INFORMATION
    # =========================================================================

    def schema_version(self) -> str | None:

        value = self.graph.get("schema_version")

        if value is None:

            metadata = self.graph.get("metadata")

            if isinstance(metadata, dict):
                value = metadata.get("schema_version")

        return _clean_string(value)

    def graph_statistics(self) -> dict[str, Any]:

        node_counts = Counter()
        relationship_counts = Counter()

        for node in self.nodes:

            node_counts[_node_type(node)] += 1

        for edge in self.edges:

            relationship_counts[_edge_type(edge)] += 1

        return {
            "schema_version": self.schema_version(),
            "nodes": len(self.nodes),
            "relationships": len(self.edges),
            "node_types": dict(
                sorted(
                    node_counts.items(),
                    key=lambda item: str(item[0]),
                )
            ),
            "relationship_types": dict(
                sorted(
                    relationship_counts.items(),
                    key=lambda item: str(item[0]),
                )
            ),
        }

    # =========================================================================
    # BASIC NODE QUERIES
    # =========================================================================

    def get_node(
        self,
        node_id: str,
    ) -> dict[str, Any] | None:

        node_id = _clean_string(node_id)

        if node_id is None:
            return None

        return self.nodes_by_id.get(node_id)

    def get_node_properties(
        self,
        node_id: str,
    ) -> dict[str, Any]:

        node = self.get_node(node_id)

        if node is None:
            return {}

        return _normalise_properties(node)

    def find_nodes_by_type(
        self,
        node_type: str,
    ) -> list[dict[str, Any]]:

        return [
            node
            for node in self.nodes
            if _node_type(node) == node_type
        ]

    # =========================================================================
    # MEMBER LOOKUP
    # =========================================================================

    def find_member(
        self,
        member_id: str,
    ) -> dict[str, Any] | None:

        member_id = _clean_string(member_id)

        if member_id is None:
            return None

        # First try direct node id.
        node = self.nodes_by_id.get(member_id)

        if node is not None:

            if _node_type(node) == MEMBER_NODE_TYPE:
                return node

        # Fall back to member_id property.
        for node in self.find_nodes_by_type(MEMBER_NODE_TYPE):

            properties = _normalise_properties(node)

            candidate = _first_non_null(
                properties,
                (
                    "member_id",
                    "patient_id",
                    "id",
                ),
            )

            if _clean_string(candidate) == member_id:
                return node

        return None

    # =========================================================================
    # MEMBER IDENTIFIER
    # =========================================================================

    def _member_identifier(
        self,
        member_node: dict[str, Any],
    ) -> str | None:

        properties = _normalise_properties(member_node)

        value = _first_non_null(
            properties,
            (
                "member_id",
                "patient_id",
            ),
        )

        if value is None:
            value = _node_id(member_node)

        return _clean_string(value)

    # =========================================================================
    # RELATIONSHIP TRAVERSAL
    # =========================================================================

    def _related_nodes(
        self,
        node_id: str,
        relationship_type: str,
        direction: str = "out",
    ) -> list[dict[str, Any]]:

        result = []

        if direction == "out":

            edges = self.outgoing.get(node_id, [])

            for edge in edges:

                if _edge_type(edge) != relationship_type:
                    continue

                target = _edge_target(edge)

                if target is None:
                    continue

                node = self.nodes_by_id.get(target)

                if node is not None:
                    result.append(node)

        elif direction == "in":

            edges = self.incoming.get(node_id, [])

            for edge in edges:

                if _edge_type(edge) != relationship_type:
                    continue

                source = _edge_source(edge)

                if source is None:
                    continue

                node = self.nodes_by_id.get(source)

                if node is not None:
                    result.append(node)

        else:

            raise ValueError(
                "direction must be 'out' or 'in'"
            )

        return result

    # =========================================================================
    # RISK
    # =========================================================================

    def get_member_risk(
        self,
        member_id: str,
    ) -> dict[str, Any] | None:

        member = self.find_member(member_id)

        if member is None:
            return None

        member_node_id = _node_id(member)

        if member_node_id is None:
            return None

        risk_nodes = self._related_nodes(
            member_node_id,
            REL_HAS_RISK,
            direction="out",
        )

        if not risk_nodes:
            return None

        # Prefer the actual RiskAssessment node.
        risk_node = risk_nodes[0]

        properties = _normalise_properties(risk_node)

        risk_probability = _safe_float(
            _first_non_null(
                properties,
                (
                    "risk_probability",
                    "risk_score",
                    "probability",
                    "predicted_probability",
                ),
            )
        )

        risk_percentile = _safe_float(
            _first_non_null(
                properties,
                (
                    "risk_percentile",
                    "percentile",
                ),
            )
        )

        risk_band = _clean_string(
            _first_non_null(
                properties,
                (
                    "risk_band",
                    "band",
                ),
            )
        )

        result = {
            "member_id": self._member_identifier(member),
            "risk_probability": risk_probability,
            "risk_percentile": risk_percentile,
            "risk_band": risk_band,
            "risk_assessment_id": _node_id(risk_node),
            "properties": properties,
        }

        return result

    # =========================================================================
    # SDOH FACTORS
    # =========================================================================

    def get_member_sdoh_factors(
        self,
        member_id: str,
    ) -> list[dict[str, Any]]:

        member = self.find_member(member_id)

        if member is None:
            return []

        node_id = _node_id(member)

        if node_id is None:
            return []

        factors = self._related_nodes(
            node_id,
            REL_HAS_SDOH,
            direction="out",
        )

        result = []

        for factor in factors:

            properties = _normalise_properties(factor)

            factor_name = _clean_string(
                _first_non_null(
                    properties,
                    (
                        "factor_name",
                        "name",
                        "label",
                        "description",
                        "factor",
                    ),
                )
            )

            domain = _clean_string(
                _first_non_null(
                    properties,
                    (
                        "domain",
                        "sdoh_domain",
                        "domain_name",
                    ),
                )
            )

            need_score = _safe_float(
                _first_non_null(
                    properties,
                    (
                        "need_score",
                        "sdoh_need_score",
                        "score",
                        "normalized_score",
                    ),
                )
            )

            severity = _clean_string(
                _first_non_null(
                    properties,
                    (
                        "severity",
                        "factor_level",
                        "need_level",
                    ),
                )
            )

            result.append(
                {
                    "factor_id": _node_id(factor),
                    "factor_name": factor_name,
                    "domain": domain,
                    "need_score": need_score,
                    "severity": severity,
                    "properties": properties,
                }
            )

        result.sort(
            key=lambda item: (
                -(item["need_score"] or 0.0),
                item["factor_name"] or "",
            )
        )

        return result

    # =========================================================================
    # SDOH DOMAINS
    # =========================================================================

    def get_member_sdoh_domains(
        self,
        member_id: str,
    ) -> list[dict[str, Any]]:

        factors = self.get_member_sdoh_factors(member_id)

        domain_map: dict[str, dict[str, Any]] = {}

        # First collect domains from factor properties.
        for factor in factors:

            domain_name = factor.get("domain")

            if not domain_name:
                continue

            entry = domain_map.setdefault(
                domain_name,
                {
                    "domain_name": domain_name,
                    "factor_count": 0,
                    "factors": [],
                    "max_need_score": None,
                    "mean_need_score": None,
                },
            )

            entry["factor_count"] += 1

            if factor["factor_name"]:
                entry["factors"].append(
                    factor["factor_name"]
                )

            score = factor.get("need_score")

            if score is not None:

                current = entry["max_need_score"]

                if current is None or score > current:
                    entry["max_need_score"] = score

        # Also traverse factor -> domain relationships.
        member = self.find_member(member_id)

        if member is not None:

            member_node_id = _node_id(member)

            if member_node_id is not None:

                factors_nodes = self._related_nodes(
                    member_node_id,
                    REL_HAS_SDOH,
                    direction="out",
                )

                for factor_node in factors_nodes:

                    factor_id = _node_id(factor_node)

                    if factor_id is None:
                        continue

                    domain_nodes = self._related_nodes(
                        factor_id,
                        REL_BELONGS_TO_DOMAIN,
                        direction="out",
                    )

                    for domain_node in domain_nodes:

                        properties = _normalise_properties(
                            domain_node
                        )

                        domain_name = _clean_string(
                            _first_non_null(
                                properties,
                                (
                                    "domain_name",
                                    "name",
                                    "label",
                                    "domain",
                                ),
                            )
                        )

                        if not domain_name:
                            continue

                        domain_map.setdefault(
                            domain_name,
                            {
                                "domain_name": domain_name,
                                "factor_count": 0,
                                "factors": [],
                                "max_need_score": None,
                                "mean_need_score": None,
                            },
                        )

        # Calculate means.
        for entry in domain_map.values():

            scores = []

            for factor in factors:

                if factor.get("domain") == entry["domain_name"]:

                    score = factor.get("need_score")

                    if score is not None:
                        scores.append(score)

            if scores:
                entry["mean_need_score"] = (
                    sum(scores) / len(scores)
                )

        result = list(domain_map.values())

        result.sort(
            key=lambda item: (
                -(item["max_need_score"] or 0.0),
                item["domain_name"] or "",
            )
        )

        return result

    # =========================================================================
    # CLINICAL FACTORS
    # =========================================================================

    def get_member_clinical_factors(
        self,
        member_id: str,
    ) -> list[dict[str, Any]]:

        member = self.find_member(member_id)

        if member is None:
            return []

        node_id = _node_id(member)

        if node_id is None:
            return []

        clinical_nodes = self._related_nodes(
            node_id,
            REL_HAS_CLINICAL,
            direction="out",
        )

        result = []

        for clinical in clinical_nodes:

            properties = _normalise_properties(clinical)

            name = _clean_string(
                _first_non_null(
                    properties,
                    (
                        "factor_name",
                        "clinical_factor",
                        "name",
                        "description",
                        "label",
                        "code",
                    ),
                )
            )

            category = _clean_string(
                _first_non_null(
                    properties,
                    (
                        "category",
                        "clinical_category",
                        "type",
                    ),
                )
            )

            value = _first_non_null(
                properties,
                (
                    "value",
                    "count",
                    "frequency",
                    "score",
                ),
            )

            result.append(
                {
                    "clinical_factor_id": _node_id(clinical),
                    "factor_name": name,
                    "category": category,
                    "value": value,
                    "properties": properties,
                }
            )

        result.sort(
            key=lambda item: item["factor_name"] or ""
        )

        return result

    # =========================================================================
    # COUNTY
    # =========================================================================

    def get_member_county(
        self,
        member_id: str,
    ) -> dict[str, Any] | None:

        member = self.find_member(member_id)

        if member is None:
            return None

        node_id = _node_id(member)

        if node_id is None:
            return None

        counties = self._related_nodes(
            node_id,
            REL_LIVES_IN,
            direction="out",
        )

        if not counties:
            return None

        county = counties[0]

        properties = _normalise_properties(county)

        county_fips = _first_non_null(
            properties,
            (
                "county_fips",
                "fips",
                "county_geoid",
                "geoid",
                "id",
            ),
        )

        county_name = _clean_string(
            _first_non_null(
                properties,
                (
                    "county_name",
                    "name",
                    "county",
                    "label",
                ),
            )
        )

        state = _clean_string(
            _first_non_null(
                properties,
                (
                    "state",
                    "state_name",
                    "state_abbr",
                ),
            )
        )

        return {
            "county_id": _node_id(county),
            "county_fips": county_fips,
            "county_name": county_name,
            "state": state,
            "properties": properties,
        }

    # =========================================================================
    # COUNTY RISK
    # =========================================================================

    def get_county_risk(
        self,
        member_id: str,
    ) -> dict[str, Any] | None:

        county = self.get_member_county(member_id)

        if county is None:
            return None

        county_id = county.get("county_id")

        if county_id is None:
            return None

        risk_nodes = self._related_nodes(
            county_id,
            REL_HAS_RISK,
            direction="out",
        )

        if not risk_nodes:
            return None

        risk_node = risk_nodes[0]

        properties = _normalise_properties(risk_node)

        return {
            "county_fips": county.get("county_fips"),
            "county_risk_probability": _safe_float(
                _first_non_null(
                    properties,
                    (
                        "county_risk_probability",
                        "risk_probability",
                        "risk_score",
                    ),
                )
            ),
            "county_risk_percentile": _safe_float(
                _first_non_null(
                    properties,
                    (
                        "county_risk_percentile",
                        "risk_percentile",
                        "percentile",
                    ),
                )
            ),
            "risk_band": _clean_string(
                _first_non_null(
                    properties,
                    (
                        "risk_band",
                        "band",
                    ),
                )
            ),
            "properties": properties,
        }

    # =========================================================================
    # INTERVENTIONS
    # =========================================================================

    def get_member_interventions(
        self,
        member_id: str,
    ) -> list[dict[str, Any]]:

        member = self.find_member(member_id)

        if member is None:
            return []

        node_id = _node_id(member)

        if node_id is None:
            return []

        intervention_nodes = self._related_nodes(
            node_id,
            REL_RECEIVES_INTERVENTION,
            direction="out",
        )

        result = []

        for intervention in intervention_nodes:

            properties = _normalise_properties(intervention)

            name = _clean_string(
                _first_non_null(
                    properties,
                    (
                        "intervention_name",
                        "name",
                        "label",
                        "recommended_intervention",
                        "intervention",
                    ),
                )
            )

            domain = _clean_string(
                _first_non_null(
                    properties,
                    (
                        "domain",
                        "sdoh_domain",
                        "domain_name",
                    ),
                )
            )

            description = _clean_string(
                _first_non_null(
                    properties,
                    (
                        "description",
                        "recommendation",
                        "action",
                    ),
                )
            )

            result.append(
                {
                    "intervention_id": _node_id(intervention),
                    "intervention_name": name,
                    "domain": domain,
                    "description": description,
                    "properties": properties,
                }
            )

        return result

    # =========================================================================
    # EVIDENCE
    # =========================================================================

    def get_member_evidence(
        self,
        member_id: str,
    ) -> list[dict[str, Any]]:

        member = self.find_member(member_id)

        if member is None:
            return []

        node_id = _node_id(member)

        if node_id is None:
            return []

        # Evidence can be connected directly or indirectly through
        # SDOH factors / interventions.
        evidence_nodes: dict[str, dict[str, Any]] = {}

        # Direct evidence.
        direct_edges = self.outgoing.get(node_id, [])

        for edge in direct_edges:

            if _edge_type(edge) != REL_SUPPORTED_BY:
                continue

            target = _edge_target(edge)

            if target is None:
                continue

            evidence = self.nodes_by_id.get(target)

            if evidence is None:
                continue

            if _node_type(evidence) != EVIDENCE_NODE_TYPE:
                continue

            evidence_nodes[target] = evidence

        # Evidence through SDOH factors.
        factors = self.get_member_sdoh_factors(member_id)

        for factor in factors:

            factor_id = factor.get("factor_id")

            if factor_id is None:
                continue

            for edge in self.outgoing.get(factor_id, []):

                if _edge_type(edge) != REL_SUPPORTED_BY:
                    continue

                target = _edge_target(edge)

                if target is None:
                    continue

                evidence = self.nodes_by_id.get(target)

                if evidence is None:
                    continue

                if _node_type(evidence) != EVIDENCE_NODE_TYPE:
                    continue

                evidence_nodes[target] = evidence

        result = []

        for evidence in evidence_nodes.values():

            properties = _normalise_properties(evidence)

            result.append(
                {
                    "evidence_id": _node_id(evidence),
                    "evidence_type": _clean_string(
                        _first_non_null(
                            properties,
                            (
                                "evidence_type",
                                "type",
                                "category",
                            ),
                        )
                    ),
                    "source": _clean_string(
                        _first_non_null(
                            properties,
                            (
                                "source",
                                "source_name",
                                "organization",
                            ),
                        )
                    ),
                    "description": _clean_string(
                        _first_non_null(
                            properties,
                            (
                                "description",
                                "statement",
                                "text",
                            ),
                        )
                    ),
                    "properties": properties,
                }
            )

        result.sort(
            key=lambda item: (
                item["evidence_type"] or "",
                item["source"] or "",
            )
        )

        return result

    # =========================================================================
    # FULL MEMBER CONTEXT
    # =========================================================================

    def get_member_context(
        self,
        member_id: str,
    ) -> dict[str, Any]:

        member = self.find_member(member_id)

        if member is None:
            raise KeyError(
                f"Member not found in knowledge graph: {member_id}"
            )

        member_identifier = self._member_identifier(member)

        risk = self.get_member_risk(member_identifier)
        sdoh_factors = self.get_member_sdoh_factors(
            member_identifier
        )
        domains = self.get_member_sdoh_domains(
            member_identifier
        )
        clinical = self.get_member_clinical_factors(
            member_identifier
        )
        county = self.get_member_county(
            member_identifier
        )
        county_risk = self.get_county_risk(
            member_identifier
        )
        interventions = self.get_member_interventions(
            member_identifier
        )
        evidence = self.get_member_evidence(
            member_identifier
        )

        return {
            "member_id": member_identifier,

            "risk": risk,

            "sdoh": {
                "factor_count": len(sdoh_factors),
                "factors": sdoh_factors,
                "domain_count": len(domains),
                "domains": domains,
            },

            "clinical": {
                "factor_count": len(clinical),
                "factors": clinical,
            },

            "geography": {
                "county": county,
                "county_risk": county_risk,
            },

            "interventions": interventions,

            "evidence": evidence,
        }

    # =========================================================================
    # INTERVENTION REASONING
    # =========================================================================

    def explain_intervention(
        self,
        member_id: str,
    ) -> dict[str, Any]:

        context = self.get_member_context(member_id)

        risk = context.get("risk") or {}

        sdoh_factors = (
            context
            .get("sdoh", {})
            .get("factors", [])
        )

        domains = (
            context
            .get("sdoh", {})
            .get("domains", [])
        )

        interventions = context.get(
            "interventions",
            [],
        )

        if not interventions:

            return {
                "member_id": member_id,
                "risk_probability": risk.get(
                    "risk_probability"
                ),
                "risk_band": risk.get("risk_band"),
                "primary_domain": (
                    domains[0]["domain_name"]
                    if domains
                    else None
                ),
                "primary_factor": (
                    sdoh_factors[0]["factor_name"]
                    if sdoh_factors
                    else None
                ),
                "recommended_interventions": [],
                "reasoning": (
                    "No intervention relationship was found "
                    "for this member."
                ),
            }

        primary_domain = None

        if domains:
            primary_domain = domains[0].get(
                "domain_name"
            )

        primary_factor = None

        if sdoh_factors:
            primary_factor = sdoh_factors[0].get(
                "factor_name"
            )

        intervention_names = [
            item.get("intervention_name")
            for item in interventions
            if item.get("intervention_name")
        ]

        reasoning_parts = []

        if risk.get("risk_band"):
            reasoning_parts.append(
                f"Member is in the "
                f"{risk['risk_band']} risk band."
            )

        if risk.get("risk_probability") is not None:
            reasoning_parts.append(
                "Risk assessment is "
                f"{risk['risk_probability']:.3f}."
            )

        if primary_domain:
            reasoning_parts.append(
                f"Dominant SDOH domain is "
                f"{primary_domain}."
            )

        if primary_factor:
            reasoning_parts.append(
                f"Highest-priority SDOH factor is "
                f"{primary_factor}."
            )

        if intervention_names:
            reasoning_parts.append(
                "The knowledge graph connects these "
                "factors to the intervention: "
                + ", ".join(intervention_names)
                + "."
            )

        return {
            "member_id": member_id,
            "risk_probability": risk.get(
                "risk_probability"
            ),
            "risk_band": risk.get("risk_band"),
            "primary_domain": primary_domain,
            "primary_factor": primary_factor,
            "recommended_interventions": intervention_names,
            "reasoning": " ".join(reasoning_parts),
        }

    # =========================================================================
    # SIMILAR MEMBERS
    # =========================================================================

    def find_similar_members(
        self,
        member_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:

        target_context = self.get_member_context(
            member_id
        )

        target_domains = {
            item.get("domain_name")
            for item in target_context["sdoh"]["domains"]
            if item.get("domain_name")
        }

        target_factors = {
            item.get("factor_name")
            for item in target_context["sdoh"]["factors"]
            if item.get("factor_name")
        }

        target_risk = (
            target_context
            .get("risk", {})
            .get("risk_probability")
        )

        candidates = []

        for node in self.find_nodes_by_type(
            MEMBER_NODE_TYPE
        ):

            candidate_id = self._member_identifier(node)

            if not candidate_id:
                continue

            if candidate_id == member_id:
                continue

            candidate_context = self.get_member_context(
                candidate_id
            )

            candidate_domains = {
                item.get("domain_name")
                for item in candidate_context["sdoh"]["domains"]
                if item.get("domain_name")
            }

            candidate_factors = {
                item.get("factor_name")
                for item in candidate_context["sdoh"]["factors"]
                if item.get("factor_name")
            }

            domain_overlap = (
                len(target_domains & candidate_domains)
            )

            factor_overlap = (
                len(target_factors & candidate_factors)
            )

            candidate_risk = (
                candidate_context
                .get("risk", {})
                .get("risk_probability")
            )

            risk_similarity = 0.0

            if (
                target_risk is not None
                and candidate_risk is not None
            ):

                risk_similarity = max(
                    0.0,
                    1.0 - abs(
                        target_risk - candidate_risk
                    ),
                )

            similarity_score = (
                0.50 * min(factor_overlap / 5.0, 1.0)
                + 0.30 * min(domain_overlap / 3.0, 1.0)
                + 0.20 * risk_similarity
            )

            candidates.append(
                {
                    "member_id": candidate_id,
                    "similarity_score": similarity_score,
                    "risk_probability": candidate_risk,
                    "risk_band": (
                        candidate_context
                        .get("risk", {})
                        .get("risk_band")
                    ),
                    "shared_domains": sorted(
                        target_domains
                        & candidate_domains
                    ),
                    "shared_factors": sorted(
                        target_factors
                        & candidate_factors
                    ),
                }
            )

        candidates.sort(
            key=lambda item: item["similarity_score"],
            reverse=True,
        )

        return candidates[:limit]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_DEFAULT_QUERY: KnowledgeGraphQuery | None = None


def get_query_engine() -> KnowledgeGraphQuery:

    global _DEFAULT_QUERY

    if _DEFAULT_QUERY is None:
        _DEFAULT_QUERY = KnowledgeGraphQuery()

    return _DEFAULT_QUERY


def get_member_context(
    member_id: str,
) -> dict[str, Any]:

    return get_query_engine().get_member_context(
        member_id
    )


def explain_intervention(
    member_id: str,
) -> dict[str, Any]:

    return get_query_engine().explain_intervention(
        member_id
    )


def find_similar_members(
    member_id: str,
    limit: int = 5,
) -> list[dict[str, Any]]:

    return get_query_engine().find_similar_members(
        member_id,
        limit,
    )


# =============================================================================
# DISPLAY HELPERS
# =============================================================================


def _print_separator(char: str = "=") -> None:
    print(char * 70)


def _print_member_context(
    context: dict[str, Any],
) -> None:

    print()
    _print_separator()

    print("MEMBER KNOWLEDGE GRAPH QUERY")

    _print_separator()

    print(
        f"Member ID: {context.get('member_id')}"
    )

    risk = context.get("risk") or {}

    print(
        "Risk probability: "
        f"{risk.get('risk_probability')}"
    )

    print(
        "Risk band: "
        f"{risk.get('risk_band')}"
    )

    sdoh = context.get("sdoh") or {}

    print(
        f"SDOH factors: "
        f"{sdoh.get('factor_count', 0)}"
    )

    print(
        f"SDOH domains: "
        f"{sdoh.get('domain_count', 0)}"
    )

    clinical = context.get("clinical") or {}

    print(
        f"Clinical factors: "
        f"{clinical.get('factor_count', 0)}"
    )

    geography = context.get("geography") or {}

    county = geography.get("county")

    if county:

        print(
            "County: resolved"
        )

        print(
            f"County FIPS: "
            f"{county.get('county_fips')}"
        )

        if county.get("county_name"):

            print(
                f"County name: "
                f"{county.get('county_name')}"
            )

    else:

        print(
            "County: unresolved"
        )

    interventions = context.get(
        "interventions",
        [],
    )

    print(
        f"Interventions: "
        f"{len(interventions)}"
    )

    evidence = context.get(
        "evidence",
        [],
    )

    print(
        f"Evidence: "
        f"{len(evidence)}"
    )

    print()
    print("SDOH FACTORS")

    for factor in sdoh.get(
        "factors",
        [],
    ):

        name = factor.get("factor_name")

        domain = factor.get("domain")

        score = factor.get("need_score")

        if name is None:
            continue

        line = f"  - {name}"

        if domain:
            line += f" [{domain}]"

        if score is not None:
            line += f" score={score:.3f}"

        print(line)

    print()
    print("SDOH DOMAINS")

    for domain in sdoh.get(
        "domains",
        [],
    ):

        name = domain.get("domain_name")

        if name:
            print(
                f"  - {name}"
            )

    print()
    print("CLINICAL FACTORS")

    for factor in clinical.get(
        "factors",
        [],
    ):

        name = factor.get("factor_name")

        if name:
            print(
                f"  - {name}"
            )

    print()
    print("INTERVENTIONS")

    for intervention in interventions:

        name = intervention.get(
            "intervention_name"
        )

        if name:
            print(
                f"  - {name}"
            )

    print()
    _print_separator()


# =============================================================================
# SELF TEST
# =============================================================================


def _run_self_test() -> None:

    print()
    _print_separator()

    print(
        "HEALTHLENS KNOWLEDGE GRAPH QUERY LAYER"
    )

    _print_separator()

    print()

    print(
        "======================================================================"
    )
    print(
        "LOADING KNOWLEDGE GRAPH"
    )
    print(
        "======================================================================"
    )

    engine = KnowledgeGraphQuery()

    print(
        f"Graph:          {engine.graph_file}"
    )

    print(
        f"Schema version: {engine.schema_version()}"
    )

    print(
        f"Nodes:           {len(engine.nodes)}"
    )

    print(
        f"Relationships:   {len(engine.edges)}"
    )

    # ------------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------------

    print()
    print(
        "======================================================================"
    )
    print(
        "GRAPH STATISTICS"
    )
    print(
        "======================================================================"
    )

    stats = engine.graph_statistics()

    print()
    print("NODE TYPES")

    for key, value in stats["node_types"].items():
        print(
            f"{str(key):35s}{value}"
        )

    print()
    print("RELATIONSHIP TYPES")

    for key, value in stats[
        "relationship_types"
    ].items():

        print(
            f"{str(key):50s}{value}"
        )

    # ------------------------------------------------------------------------
    # FIND A REAL MEMBER
    # ------------------------------------------------------------------------

    members = engine.find_nodes_by_type(
        MEMBER_NODE_TYPE
    )

    if not members:

        raise AssertionError(
            "No MEMBER nodes found."
        )

    first_member = members[0]

    member_id = engine._member_identifier(
        first_member
    )

    if member_id is None:

        raise AssertionError(
            "MEMBER node does not contain member_id."
        )

    # ------------------------------------------------------------------------
    # MEMBER CONTEXT
    # ------------------------------------------------------------------------

    context = engine.get_member_context(
        member_id
    )

    _print_member_context(context)

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    print()
    print(
        "======================================================================"
    )
    print(
        "QUERY LAYER SELF-TEST"
    )
    print(
        "======================================================================"
    )

    # Graph loading.
    assert len(engine.nodes) > 0
    assert len(engine.edges) > 0

    print(
        "Graph loading:          PASS"
    )

    # Member lookup.
    assert context["member_id"] == member_id

    print(
        "Member lookup:          PASS"
    )

    # Risk query.
    risk = context.get("risk")

    assert isinstance(risk, dict)

    print(
        "Risk query:             PASS"
    )

    # SDOH query.
    sdoh = context.get("sdoh")

    assert isinstance(sdoh, dict)
    assert "factors" in sdoh

    print(
        "SDOH query:             PASS"
    )

    # Clinical query.
    clinical = context.get("clinical")

    assert isinstance(clinical, dict)
    assert "factors" in clinical

    print(
        "Clinical query:         PASS"
    )

    # Domain query.
    assert "domains" in sdoh

    print(
        "Domain query:           PASS"
    )

    # Context aggregation.
    assert (
        "geography" in context
        and "interventions" in context
        and "evidence" in context
    )

    print(
        "Context aggregation:    PASS"
    )

    # ------------------------------------------------------------------------
    # IMPORTANT PROPERTY TESTS
    # ------------------------------------------------------------------------

    print()
    print(
        "PROPERTY EXTRACTION VALIDATION"
    )

    # Member id must never be None.
    assert context["member_id"] is not None

    print(
        "Member ID extraction:   PASS"
    )

    # Validate SDOH factor extraction if factors exist.
    factors = sdoh.get(
        "factors",
        [],
    )

    if factors:

        named_factors = [
            item
            for item in factors
            if item.get("factor_name")
        ]

        assert named_factors, (
            "SDOH factors exist but none have "
            "a usable factor_name."
        )

        print(
            "SDOH name extraction:   PASS"
        )

    else:

        print(
            "SDOH name extraction:   PASS "
            "(no SDOH factors)"
        )

    # Validate interventions.
    interventions = context.get(
        "interventions",
        [],
    )

    if interventions:

        named_interventions = [
            item
            for item in interventions
            if item.get("intervention_name")
        ]

        assert named_interventions, (
            "Intervention nodes exist but no "
            "intervention names were extracted."
        )

        print(
            "Intervention extraction: PASS"
        )

    else:

        print(
            "Intervention extraction: PASS "
            "(no interventions)"
        )

    print()
    print(
        "======================================================================"
    )
    print(
        "QUERY LAYER SELF-TEST: PASSED"
    )
    print(
        "======================================================================"
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    _run_self_test()


if __name__ == "__main__":
    main()