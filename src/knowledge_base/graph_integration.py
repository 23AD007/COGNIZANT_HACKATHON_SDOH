"""
===============================================================================
HEALTHLENS KNOWLEDGE BASE -> KNOWLEDGE GRAPH INTEGRATION
===============================================================================

Purpose
-------
Integrate the validated HealthLens Knowledge Base with the existing
Knowledge Graph without changing the existing graph schema.

Architecture
------------

    Knowledge Base
    ----------------
        SDOH Domains
             |
        SDOH Factors
          /      \
     Evidence   Interventions
             |
             v
    Knowledge Graph
    ----------------
        MEMBER
          |
          +-- RiskAssessment
          |
          +-- ClinicalFactor
          |
          +-- SDOH_FACTOR
          |      |
          |      +-- SDOH_DOMAIN
          |      +-- Evidence
          |      +-- Intervention
          |
          +-- COUNTY
                 |
                 +-- CountyRiskAssessment


Important
---------
The Knowledge Base is authoritative for:

    - SDOH domains
    - SDOH factors
    - interventions
    - evidence

The Knowledge Graph is authoritative for:

    - members
    - member risk assessments
    - clinical context
    - member SDOH relationships
    - county context
    - graph relationships

The ML model remains responsible for:

    - risk probability
    - risk band
    - prediction

This module does NOT replace the ML model.
This module does NOT rebuild the existing graph.
This module does NOT mutate the existing graph.

It creates a separate integration artifact containing:

    - graph metadata
    - KB metadata
    - KB indexes
    - graph statistics
    - relationship statistics
    - factor/evidence/intervention mappings
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping
import json


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "knowledge_graph"
    / "healthlens_knowledge_graph.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "knowledge_graph"
)

INTEGRATION_OUTPUT_PATH = (
    OUTPUT_DIR
    / "healthlens_kb_graph_integration.json"
)


# =============================================================================
# VERSION
# =============================================================================

INTEGRATION_VERSION = "1.0.1"


# =============================================================================
# IMPORT REGISTRY
# =============================================================================

try:
    from .registry import build_registry
except ImportError:
    from src.knowledge_base.registry import build_registry


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def normalize_text(value: Any) -> str:
    """
    Normalize text for comparisons.

    Examples:

        Economic Stability
        economic_stability
        Economic-Stability

    all become the same comparison key.
    """

    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def canonical_id(value: Any) -> str:
    """
    Convert a value to a clean string identifier.
    """

    return str(value or "").strip()


def get_attr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Read a value from either an object or dictionary.
    """

    if isinstance(obj, Mapping):
        return obj.get(name, default)

    return getattr(obj, name, default)


def object_id(
    obj: Any,
    *,
    fallback: str = "",
) -> str:
    """
    Extract an identifier from a KB object.

    Supports the naming conventions used by the project.
    """

    attributes = (
        "id",
        "evidence_id",
        "intervention_id",
        "factor_id",
        "domain_id",
        "key",
        "code",
    )

    for attribute in attributes:

        value = get_attr(
            obj,
            attribute,
            None,
        )

        if value:
            return canonical_id(value)

    return canonical_id(fallback)


def object_name(
    obj: Any,
    *,
    fallback: str = "",
) -> str:
    """
    Extract a human-readable name.
    """

    attributes = (
        "name",
        "label",
        "display_name",
        "title",
        "factor",
        "domain",
    )

    for attribute in attributes:

        value = get_attr(
            obj,
            attribute,
            None,
        )

        if value:
            return canonical_id(value)

    return canonical_id(fallback)


def ensure_list(
    value: Any,
) -> list[Any]:
    """
    Convert common collection forms to a list.
    """

    if value is None:
        return []

    if isinstance(value, Mapping):
        return list(value.values())

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return list(value)

    return [value]


# =============================================================================
# GRAPH LOADING
# =============================================================================

def load_graph(
    graph_path: Path = GRAPH_PATH,
) -> dict[str, Any]:
    """
    Load the existing knowledge graph.

    The returned graph is never modified.
    """

    if not graph_path.exists():

        raise FileNotFoundError(
            f"Knowledge graph not found: {graph_path}"
        )

    with graph_path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        graph = json.load(handle)

    if not isinstance(
        graph,
        Mapping,
    ):

        raise ValueError(
            "Knowledge graph JSON must contain an object."
        )

    return dict(graph)


# =============================================================================
# GRAPH STRUCTURE DISCOVERY
# =============================================================================

def _first_existing_key(
    mapping: Mapping[str, Any],
    candidates: Iterable[str],
) -> str | None:
    """
    Return the first existing key from candidates.
    """

    for key in candidates:

        if key in mapping:
            return key

    return None


def extract_graph_nodes(
    graph: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract graph nodes from the existing graph structure.

    Supports:

        {
            "nodes": [...]
        }

    and:

        {
            "graph": {
                "nodes": [...]
            }
        }

    and common alternative names.
    """

    containers: list[Mapping[str, Any]] = [
        graph
    ]

    for nested_key in (
        "graph",
        "data",
        "knowledge_graph",
        "payload",
    ):

        nested = graph.get(
            nested_key
        )

        if isinstance(
            nested,
            Mapping,
        ):

            containers.append(
                nested
            )

    candidates = (
        "nodes",
        "Nodes",
        "node_list",
        "nodeList",
        "graph_nodes",
    )

    for container in containers:

        key = _first_existing_key(
            container,
            candidates,
        )

        if key is None:
            continue

        value = container[key]

        if isinstance(
            value,
            Mapping,
        ):

            result = []

            for node_id, node in value.items():

                if isinstance(
                    node,
                    Mapping,
                ):

                    item = dict(node)

                    if not any(
                        item.get(identifier)
                        for identifier in (
                            "id",
                            "node_id",
                            "nodeId",
                        )
                    ):

                        item.setdefault(
                            "id",
                            node_id,
                        )

                    result.append(
                        item
                    )

            return result

        if isinstance(
            value,
            list,
        ):

            return [
                dict(node)
                for node in value
                if isinstance(
                    node,
                    Mapping,
                )
            ]

    return []


def extract_graph_relationships(
    graph: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract relationships from the EXISTING graph.

    This function is deliberately more robust than the previous version.

    It supports:

        relationships
        edges
        graph.relationships
        graph.edges
        graph.relationships as list
        graph.relationships as dictionary
        relationship collections under data/knowledge_graph

    The function preserves relationship objects as dictionaries.

    This is the critical fix for the previous:

        Graph relationships: 0

    result.
    """

    containers: list[Mapping[str, Any]] = [
        graph
    ]

    # -------------------------------------------------------------------------
    # Nested graph containers
    # -------------------------------------------------------------------------

    for nested_key in (
        "graph",
        "data",
        "knowledge_graph",
        "payload",
    ):

        nested = graph.get(
            nested_key
        )

        if isinstance(
            nested,
            Mapping,
        ):

            containers.append(
                nested
            )

    # -------------------------------------------------------------------------
    # Candidate relationship keys
    # -------------------------------------------------------------------------

    candidates = (
        "relationships",
        "Relationships",
        "relationship",
        "edges",
        "Edges",
        "edge_list",
        "edgeList",
        "graph_relationships",
        "graph_edges",
    )

    for container in containers:

        key = _first_existing_key(
            container,
            candidates,
        )

        if key is None:
            continue

        value = container[key]

        # ---------------------------------------------------------------------
        # Dictionary-backed relationships
        # ---------------------------------------------------------------------

        if isinstance(
            value,
            Mapping,
        ):

            result: list[dict[str, Any]] = []

            for relationship_id, relationship in (
                value.items()
            ):

                # -------------------------------------------------------------
                # Standard relationship object
                # -------------------------------------------------------------

                if isinstance(
                    relationship,
                    Mapping,
                ):

                    item = dict(
                        relationship
                    )

                    if not any(
                        item.get(identifier)
                        for identifier in (
                            "id",
                            "relationship_id",
                            "relationshipId",
                            "edge_id",
                            "edgeId",
                        )
                    ):

                        item.setdefault(
                            "id",
                            relationship_id,
                        )

                    result.append(
                        item
                    )

                # -------------------------------------------------------------
                # Some serialized graphs may use:
                #
                # "relationship_id": ["source", "target"]
                # -------------------------------------------------------------

                elif isinstance(
                    relationship,
                    (list, tuple),
                ):

                    item = {
                        "id": relationship_id,
                        "value": list(
                            relationship
                        ),
                    }

                    result.append(
                        item
                    )

            return result

        # ---------------------------------------------------------------------
        # List-backed relationships
        # ---------------------------------------------------------------------

        if isinstance(
            value,
            list,
        ):

            return [
                dict(relationship)
                for relationship in value
                if isinstance(
                    relationship,
                    Mapping,
                )
            ]

    # -------------------------------------------------------------------------
    # Fallback:
    #
    # Some graph serializers use "links".
    # -------------------------------------------------------------------------

    for container in containers:

        if "links" not in container:
            continue

        value = container["links"]

        if isinstance(
            value,
            list,
        ):

            return [
                dict(relationship)
                for relationship in value
                if isinstance(
                    relationship,
                    Mapping,
                )
            ]

        if isinstance(
            value,
            Mapping,
        ):

            result = []

            for relationship_id, relationship in (
                value.items()
            ):

                if isinstance(
                    relationship,
                    Mapping,
                ):

                    item = dict(
                        relationship
                    )

                    item.setdefault(
                        "id",
                        relationship_id,
                    )

                    result.append(
                        item
                    )

            return result

    return []


# =============================================================================
# GRAPH STATISTICS
# =============================================================================

def graph_node_count(
    graph: Mapping[str, Any],
) -> int:
    """
    Count graph nodes.
    """

    return len(
        extract_graph_nodes(
            graph
        )
    )


def graph_relationship_count(
    graph: Mapping[str, Any],
) -> int:
    """
    Count graph relationships.

    IMPORTANT:
    This uses the actual relationship/edge collection and does not assume
    that relationships must be stored under one particular key.
    """

    return len(
        extract_graph_relationships(
            graph
        )
    )


def relationship_type_counts(
    graph: Mapping[str, Any],
) -> dict[str, int]:
    """
    Count relationships by type.

    Supports common relationship type property names.
    """

    relationships = (
        extract_graph_relationships(
            graph
        )
    )

    counts: dict[str, int] = {}

    for relationship in relationships:

        relationship_type = None

        for key in (
            "type",
            "relationship_type",
            "relationshipType",
            "rel_type",
            "relType",
        ):

            value = relationship.get(
                key
            )

            if value:
                relationship_type = (
                    str(value)
                )
                break

        if relationship_type is None:
            relationship_type = "UNKNOWN"

        counts[
            relationship_type
        ] = (
            counts.get(
                relationship_type,
                0,
            )
            + 1
        )

    return counts


# =============================================================================
# REGISTRY EXTRACTION
# =============================================================================

def extract_registry_collection(
    registry: Any,
    names: Iterable[str],
) -> list[Any]:
    """
    Extract a collection from the registry.

    Supports both:

        registry.factors

    and:

        registry.get_all_factors()

    and other project-compatible forms.
    """

    for name in names:

        if not hasattr(
            registry,
            name,
        ):

            continue

        value = getattr(
            registry,
            name,
        )

        try:

            value = (
                value()
                if callable(value)
                else value
            )

        except TypeError:

            continue

        if value is None:
            continue

        return ensure_list(
            value
        )

    return []


def load_registry() -> Any:
    """
    Build the validated Knowledge Base registry.
    """

    registry = build_registry()

    if registry is None:

        raise RuntimeError(
            "Knowledge Base registry could not be built."
        )

    return registry


# =============================================================================
# REGISTRY COLLECTIONS
# =============================================================================

def get_domains(
    registry: Any,
) -> list[Any]:
    """
    Retrieve all SDOH domains.
    """

    return extract_registry_collection(
        registry,
        (
            "domains",
            "sdoh_domains",
            "get_all_domains",
            "all_domains",
        ),
    )


def get_factors(
    registry: Any,
) -> list[Any]:
    """
    Retrieve all SDOH factors.
    """

    return extract_registry_collection(
        registry,
        (
            "factors",
            "sdoh_factors",
            "get_all_factors",
            "all_factors",
        ),
    )


def get_interventions(
    registry: Any,
) -> list[Any]:
    """
    Retrieve all interventions.
    """

    return extract_registry_collection(
        registry,
        (
            "interventions",
            "get_all_interventions",
            "all_interventions",
        ),
    )


def get_evidence(
    registry: Any,
) -> list[Any]:
    """
    Retrieve all evidence records.

    Evidence is intentionally treated as a list.

    This is compatible with the current evidence.py implementation.
    """

    return extract_registry_collection(
        registry,
        (
            "evidence",
            "evidence_records",
            "get_all_evidence",
            "all_evidence",
        ),
    )


# =============================================================================
# INDEX CONTAINER
# =============================================================================

@dataclass
class KnowledgeBaseIndexes:
    """
    Runtime indexes for KB-aware graph queries.
    """

    domains: dict[str, Any] = field(
        default_factory=dict
    )

    factors: dict[str, Any] = field(
        default_factory=dict
    )

    interventions: dict[str, Any] = field(
        default_factory=dict
    )

    evidence: dict[str, Any] = field(
        default_factory=dict
    )

    factor_to_evidence: dict[
        str,
        list[str],
    ] = field(
        default_factory=dict
    )

    intervention_to_evidence: dict[
        str,
        list[str],
    ] = field(
        default_factory=dict
    )

    factor_to_interventions: dict[
        str,
        list[str],
    ] = field(
        default_factory=dict
    )

    domain_to_factors: dict[
        str,
        list[str],
    ] = field(
        default_factory=dict
    )

    def as_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert indexes to serializable form.

        KB objects are converted with asdict when possible and otherwise
        represented using their __dict__.
        """

        def serialize(
            value: Any,
        ) -> Any:

            if isinstance(
                value,
                Mapping,
            ):

                return {
                    str(key): serialize(item)
                    for key, item in value.items()
                }

            if isinstance(
                value,
                list,
            ):

                return [
                    serialize(item)
                    for item in value
                ]

            try:

                return asdict(
                    value
                )

            except TypeError:

                if hasattr(
                    value,
                    "__dict__",
                ):

                    return dict(
                        value.__dict__
                    )

                return value

        return serialize(
            asdict(self)
        )


# =============================================================================
# INDEX BUILDING
# =============================================================================

def build_indexes(
    registry: Any,
) -> KnowledgeBaseIndexes:
    """
    Build all KB indexes.

    Evidence is explicitly indexed even though evidence.py stores records
    in a list.
    """

    domains = get_domains(
        registry
    )

    factors = get_factors(
        registry
    )

    interventions = get_interventions(
        registry
    )

    evidence_records = get_evidence(
        registry
    )

    indexes = KnowledgeBaseIndexes()

    # =========================================================================
    # DOMAINS
    # =========================================================================

    for domain in domains:

        domain_id = object_id(
            domain,
            fallback=object_name(
                domain
            ),
        )

        if not domain_id:
            continue

        indexes.domains[
            domain_id
        ] = domain

    # =========================================================================
    # FACTORS
    # =========================================================================

    for factor in factors:

        factor_id = object_id(
            factor,
            fallback=object_name(
                factor
            ),
        )

        if not factor_id:
            continue

        indexes.factors[
            factor_id
        ] = factor

        domain_value = get_attr(
            factor,
            "domain",
            None,
        )

        if domain_value:

            domain_key = normalize_text(
                domain_value
            )

            indexes.domain_to_factors.setdefault(
                domain_key,
                [],
            ).append(
                factor_id
            )

    # =========================================================================
    # INTERVENTIONS
    # =========================================================================

    for intervention in interventions:

        intervention_id = object_id(
            intervention,
            fallback=object_name(
                intervention
            ),
        )

        if not intervention_id:
            continue

        indexes.interventions[
            intervention_id
        ] = intervention

        target_factors = get_attr(
            intervention,
            "target_factors",
            None,
        )

        if target_factors is None:

            target_factors = get_attr(
                intervention,
                "factors",
                [],
            )

        for target_factor in ensure_list(
            target_factors
        ):

            target_factor_id = canonical_id(
                target_factor
            )

            if not target_factor_id:
                continue

            indexes.factor_to_interventions.setdefault(
                target_factor_id,
                [],
            ).append(
                intervention_id
            )

    # =========================================================================
    # EVIDENCE
    # =========================================================================

    for position, evidence in enumerate(
        evidence_records
    ):

        evidence_id = object_id(
            evidence,
            fallback=f"evidence_{position + 1}",
        )

        if not evidence_id:

            evidence_id = (
                f"evidence_{position + 1}"
            )

        indexes.evidence[
            evidence_id
        ] = evidence

        # ---------------------------------------------------------------------
        # FACTOR -> EVIDENCE
        # ---------------------------------------------------------------------

        factor_value = get_attr(
            evidence,
            "factor",
            None,
        )

        if factor_value:

            factor_id = canonical_id(
                factor_value
            )

            indexes.factor_to_evidence.setdefault(
                factor_id,
                [],
            ).append(
                evidence_id
            )

        # ---------------------------------------------------------------------
        # INTERVENTION -> EVIDENCE
        # ---------------------------------------------------------------------

        intervention_value = get_attr(
            evidence,
            "intervention_id",
            None,
        )

        if intervention_value:

            intervention_id = canonical_id(
                intervention_value
            )

            indexes.intervention_to_evidence.setdefault(
                intervention_id,
                [],
            ).append(
                evidence_id
            )

    return indexes


# =============================================================================
# INDEX VALIDATION
# =============================================================================

def validate_indexes(
    indexes: KnowledgeBaseIndexes,
    *,
    expected_evidence_count: int | None = None,
) -> None:
    """
    Validate all indexes.
    """

    if not indexes.domains:

        raise AssertionError(
            "Domain index is empty."
        )

    if not indexes.factors:

        raise AssertionError(
            "Factor index is empty."
        )

    if not indexes.interventions:

        raise AssertionError(
            "Intervention index is empty."
        )

    if (
        expected_evidence_count is not None
        and expected_evidence_count > 0
    ):

        if not indexes.evidence:

            raise AssertionError(
                "Evidence index is empty."
            )

        if len(
            indexes.evidence
        ) != expected_evidence_count:

            raise AssertionError(
                "Evidence index count mismatch: "
                f"expected {expected_evidence_count}, "
                f"got {len(indexes.evidence)}."
            )

    # =========================================================================
    # FACTOR -> EVIDENCE
    # =========================================================================

    for (
        factor_id,
        evidence_ids,
    ) in indexes.factor_to_evidence.items():

        for evidence_id in evidence_ids:

            if evidence_id not in (
                indexes.evidence
            ):

                raise AssertionError(
                    "Factor -> evidence index references "
                    f"unknown evidence '{evidence_id}'."
                )

    # =========================================================================
    # INTERVENTION -> EVIDENCE
    # =========================================================================

    for (
        intervention_id,
        evidence_ids,
    ) in indexes.intervention_to_evidence.items():

        for evidence_id in evidence_ids:

            if evidence_id not in (
                indexes.evidence
            ):

                raise AssertionError(
                    "Intervention -> evidence index references "
                    f"unknown evidence '{evidence_id}'."
                )

    # =========================================================================
    # FACTOR -> INTERVENTION
    # =========================================================================

    for (
        factor_id,
        intervention_ids,
    ) in indexes.factor_to_interventions.items():

        for intervention_id in intervention_ids:

            if intervention_id not in (
                indexes.interventions
            ):

                raise AssertionError(
                    "Factor -> intervention index references "
                    f"unknown intervention '{intervention_id}'."
                )


# =============================================================================
# REGISTRY METADATA
# =============================================================================

def registry_version(
    registry: Any,
) -> str:
    """
    Retrieve registry version.
    """

    for attribute in (
        "registry_version",
        "version",
        "knowledge_base_version",
    ):

        value = getattr(
            registry,
            attribute,
            None,
        )

        if value:
            return str(value)

    return "unknown"


def graph_schema_version(
    graph: Mapping[str, Any],
) -> str:
    """
    Retrieve graph schema version.
    """

    for key in (
        "schema_version",
        "version",
    ):

        value = graph.get(
            key
        )

        if value:
            return str(value)

    metadata = graph.get(
        "metadata",
        {},
    )

    if isinstance(
        metadata,
        Mapping,
    ):

        value = metadata.get(
            "schema_version"
        )

        if value:
            return str(value)

    return "unknown"


# =============================================================================
# GRAPH TYPE STATISTICS
# =============================================================================

def graph_node_type_counts(
    graph: Mapping[str, Any],
) -> dict[str, int]:
    """
    Count graph nodes by node type.
    """

    nodes = extract_graph_nodes(
        graph
    )

    counts: dict[str, int] = {}

    for node in nodes:

        node_type = None

        for key in (
            "type",
            "node_type",
            "nodeType",
            "label",
            "kind",
        ):

            value = node.get(
                key
            )

            if value:
                node_type = str(
                    value
                )
                break

        if node_type is None:
            node_type = "UNKNOWN"

        counts[
            node_type
        ] = (
            counts.get(
                node_type,
                0,
            )
            + 1
        )

    return counts


# =============================================================================
# INTEGRATION RESULT
# =============================================================================

@dataclass
class GraphIntegrationResult:
    """
    Complete integration result.
    """

    integration_version: str

    graph_path: str

    registry_version: str

    schema_version: str

    node_count: int

    relationship_count: int

    node_type_counts: dict[str, int]

    relationship_type_counts: dict[str, int]

    domain_count: int

    factor_count: int

    intervention_count: int

    evidence_count: int

    indexes: KnowledgeBaseIndexes

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "integration_version": (
                self.integration_version
            ),
            "graph_path": (
                self.graph_path
            ),
            "registry_version": (
                self.registry_version
            ),
            "schema_version": (
                self.schema_version
            ),
            "graph": {
                "node_count": (
                    self.node_count
                ),
                "relationship_count": (
                    self.relationship_count
                ),
                "node_type_counts": (
                    self.node_type_counts
                ),
                "relationship_type_counts": (
                    self.relationship_type_counts
                ),
            },
            "knowledge_base": {
                "domain_count": (
                    self.domain_count
                ),
                "factor_count": (
                    self.factor_count
                ),
                "intervention_count": (
                    self.intervention_count
                ),
                "evidence_count": (
                    self.evidence_count
                ),
            },
            "indexes": (
                self.indexes.as_dict()
            ),
        }


# =============================================================================
# INTEGRATION
# =============================================================================

def integrate_knowledge_base_with_graph(
    *,
    graph_path: Path = GRAPH_PATH,
    save: bool = True,
) -> GraphIntegrationResult:
    """
    Integrate the validated Knowledge Base with the existing graph.

    No existing graph nodes or relationships are changed.
    """

    graph = load_graph(
        graph_path
    )

    registry = load_registry()

    domains = get_domains(
        registry
    )

    factors = get_factors(
        registry
    )

    interventions = get_interventions(
        registry
    )

    evidence_records = get_evidence(
        registry
    )

    indexes = build_indexes(
        registry
    )

    validate_indexes(
        indexes,
        expected_evidence_count=len(
            evidence_records
        ),
    )

    nodes = extract_graph_nodes(
        graph
    )

    relationships = extract_graph_relationships(
        graph
    )

    result = GraphIntegrationResult(
        integration_version=(
            INTEGRATION_VERSION
        ),
        graph_path=str(
            graph_path
        ),
        registry_version=(
            registry_version(
                registry
            )
        ),
        schema_version=(
            graph_schema_version(
                graph
            )
        ),
        node_count=len(
            nodes
        ),
        relationship_count=len(
            relationships
        ),
        node_type_counts=(
            graph_node_type_counts(
                graph
            )
        ),
        relationship_type_counts=(
            relationship_type_counts(
                graph
            )
        ),
        domain_count=len(
            domains
        ),
        factor_count=len(
            factors
        ),
        intervention_count=len(
            interventions
        ),
        evidence_count=len(
            evidence_records
        ),
        indexes=indexes,
    )

    if save:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with INTEGRATION_OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                result.to_dict(),
                handle,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    return result


# =============================================================================
# QUERY HELPERS
# =============================================================================

def get_indexed_evidence(
    indexes: KnowledgeBaseIndexes,
    evidence_id: str,
) -> Any | None:
    """
    Retrieve evidence by ID.
    """

    return indexes.evidence.get(
        canonical_id(
            evidence_id
        )
    )


def get_evidence_for_factor(
    indexes: KnowledgeBaseIndexes,
    factor_id: str,
) -> list[Any]:
    """
    Retrieve evidence for a factor.
    """

    evidence_ids = (
        indexes.factor_to_evidence.get(
            canonical_id(
                factor_id
            ),
            [],
        )
    )

    return [
        indexes.evidence[evidence_id]
        for evidence_id in evidence_ids
        if evidence_id in indexes.evidence
    ]


def get_interventions_for_factor(
    indexes: KnowledgeBaseIndexes,
    factor_id: str,
) -> list[Any]:
    """
    Retrieve interventions targeting a factor.
    """

    intervention_ids = (
        indexes.factor_to_interventions.get(
            canonical_id(
                factor_id
            ),
            [],
        )
    )

    return [
        indexes.interventions[
            intervention_id
        ]
        for intervention_id in intervention_ids
        if intervention_id in indexes.interventions
    ]


def get_factors_for_domain(
    indexes: KnowledgeBaseIndexes,
    domain: str,
) -> list[Any]:
    """
    Retrieve factors belonging to a domain.
    """

    domain_key = normalize_text(
        domain
    )

    factor_ids = (
        indexes.domain_to_factors.get(
            domain_key,
            [],
        )
    )

    return [
        indexes.factors[
            factor_id
        ]
        for factor_id in factor_ids
        if factor_id in indexes.factors
    ]


# =============================================================================
# SELF TEST
# =============================================================================

def _self_test() -> None:

    print("=" * 78)
    print(
        "HEALTHLENS KNOWLEDGE BASE → GRAPH INTEGRATION SELF-TEST"
    )
    print("=" * 78)

    # =========================================================================
    # GRAPH
    # =========================================================================

    graph = load_graph()

    print(
        "Graph loading:              PASS"
    )

    # =========================================================================
    # GRAPH COUNTS
    # =========================================================================

    nodes = extract_graph_nodes(
        graph
    )

    relationships = extract_graph_relationships(
        graph
    )

    node_count = len(
        nodes
    )

    relationship_count = len(
        relationships
    )

    print(
        f"Graph nodes detected:       {node_count}"
    )

    print(
        f"Graph relationships detected: {relationship_count}"
    )

    # -------------------------------------------------------------------------
    # IMPORTANT SAFETY CHECK
    #
    # The existing HealthLens graph is expected to contain the relationships
    # created by the earlier graph builder.
    #
    # The known graph generated by the current workflow contains 3407.
    # If the actual JSON contains zero, stop here instead of silently passing.
    # -------------------------------------------------------------------------

    if relationship_count == 0:

        raise AssertionError(
            "Existing knowledge graph contains zero detected "
            "relationships. The graph structure could not be resolved. "
            "Do not proceed until the graph relationship container is "
            "identified."
        )

    print(
        "Graph relationship extraction: PASS"
    )

    # =========================================================================
    # REGISTRY
    # =========================================================================

    registry = load_registry()

    print(
        "Registry loading:           PASS"
    )

    # =========================================================================
    # COLLECTIONS
    # =========================================================================

    domains = get_domains(
        registry
    )

    factors = get_factors(
        registry
    )

    interventions = get_interventions(
        registry
    )

    evidence_records = get_evidence(
        registry
    )

    assert domains
    assert factors
    assert interventions
    assert evidence_records

    print(
        "Knowledge collections:      PASS"
    )

    # =========================================================================
    # EXPECTED CURRENT COUNTS
    # =========================================================================

    assert len(
        domains
    ) == 8, (
        "Expected 8 SDOH domains, "
        f"found {len(domains)}."
    )

    assert len(
        factors
    ) == 43, (
        "Expected 43 SDOH factors, "
        f"found {len(factors)}."
    )

    assert len(
        interventions
    ) == 8, (
        "Expected 8 interventions, "
        f"found {len(interventions)}."
    )

    assert len(
        evidence_records
    ) == 40, (
        "Expected 40 evidence records, "
        f"found {len(evidence_records)}."
    )

    print(
        "Knowledge base counts:      PASS"
    )

    # =========================================================================
    # INDEXES
    # =========================================================================

    indexes = build_indexes(
        registry
    )

    validate_indexes(
        indexes,
        expected_evidence_count=len(
            evidence_records
        ),
    )

    print(
        "Index construction:         PASS"
    )

    # =========================================================================
    # DOMAIN INDEX
    # =========================================================================

    assert indexes.domains

    assert len(
        indexes.domains
    ) == len(
        domains
    )

    print(
        "Domain index:               PASS"
    )

    # =========================================================================
    # FACTOR INDEX
    # =========================================================================

    assert indexes.factors

    assert len(
        indexes.factors
    ) == len(
        factors
    )

    print(
        "Factor index:               PASS"
    )

    # =========================================================================
    # INTERVENTION INDEX
    # =========================================================================

    assert indexes.interventions

    assert len(
        indexes.interventions
    ) == len(
        interventions
    )

    print(
        "Intervention index:         PASS"
    )

    # =========================================================================
    # EVIDENCE INDEX
    # =========================================================================

    assert indexes.evidence

    assert len(
        indexes.evidence
    ) == len(
        evidence_records
    )

    print(
        "Evidence index:             PASS"
    )

    # =========================================================================
    # FACTOR -> EVIDENCE
    # =========================================================================

    assert indexes.factor_to_evidence

    print(
        "Factor → evidence index:    PASS"
    )

    # =========================================================================
    # FACTOR -> INTERVENTION
    # =========================================================================

    assert indexes.factor_to_interventions

    print(
        "Factor → intervention index: PASS"
    )

    # =========================================================================
    # EVIDENCE RETRIEVAL
    # =========================================================================

    first_evidence_id = next(
        iter(
            indexes.evidence
        )
    )

    first_evidence = (
        get_indexed_evidence(
            indexes,
            first_evidence_id,
        )
    )

    assert first_evidence is not None

    print(
        "Evidence retrieval:         PASS"
    )

    # =========================================================================
    # FACTOR EVIDENCE QUERY
    # =========================================================================

    first_factor_id = next(
        iter(
            indexes.factors
        )
    )

    factor_evidence = (
        get_evidence_for_factor(
            indexes,
            first_factor_id,
        )
    )

    assert isinstance(
        factor_evidence,
        list,
    )

    print(
        "Factor evidence query:      PASS"
    )

    # =========================================================================
    # INTERVENTION QUERY
    # =========================================================================

    intervention_results = (
        get_interventions_for_factor(
            indexes,
            first_factor_id,
        )
    )

    assert isinstance(
        intervention_results,
        list,
    )

    print(
        "Intervention query:         PASS"
    )

    # =========================================================================
    # DOMAIN QUERY
    # =========================================================================

    first_domain_id = next(
        iter(
            indexes.domains
        )
    )

    domain_factors = (
        get_factors_for_domain(
            indexes,
            first_domain_id,
        )
    )

    assert isinstance(
        domain_factors,
        list,
    )

    print(
        "Domain → factor query:      PASS"
    )

    # =========================================================================
    # FULL INTEGRATION
    # =========================================================================

    result = (
        integrate_knowledge_base_with_graph(
            save=True
        )
    )

    assert result.node_count == node_count

    assert result.relationship_count == (
        relationship_count
    )

    assert result.domain_count == 8

    assert result.factor_count == 43

    assert result.intervention_count == 8

    assert result.evidence_count == 40

    print(
        "Full graph integration:     PASS"
    )

    # =========================================================================
    # OUTPUT VALIDATION
    # =========================================================================

    assert INTEGRATION_OUTPUT_PATH.exists()

    with INTEGRATION_OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:

        saved = json.load(
            handle
        )

    assert (
        saved["graph"]["node_count"]
        == node_count
    )

    assert (
        saved["graph"]["relationship_count"]
        == relationship_count
    )

    assert (
        saved["knowledge_base"]["domain_count"]
        == 8
    )

    assert (
        saved["knowledge_base"]["factor_count"]
        == 43
    )

    assert (
        saved["knowledge_base"]["intervention_count"]
        == 8
    )

    assert (
        saved["knowledge_base"]["evidence_count"]
        == 40
    )

    assert saved[
        "indexes"
    ]["evidence"]

    print(
        "Integration artifact:      PASS"
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print(
        "INTEGRATION SUMMARY"
    )
    print("-" * 78)

    print(
        f"Domains:                    "
        f"{result.domain_count}"
    )

    print(
        f"Factors:                    "
        f"{result.factor_count}"
    )

    print(
        f"Interventions:              "
        f"{result.intervention_count}"
    )

    print(
        f"Evidence records:           "
        f"{result.evidence_count}"
    )

    print(
        f"Graph nodes:                "
        f"{result.node_count}"
    )

    print(
        f"Graph relationships:        "
        f"{result.relationship_count}"
    )

    # =========================================================================
    # RELATIONSHIP TYPES
    # =========================================================================

    print()
    print(
        "GRAPH RELATIONSHIP TYPES"
    )
    print("-" * 78)

    for (
        relationship_type,
        count,
    ) in sorted(
        result.relationship_type_counts.items()
    ):

        print(
            f"{relationship_type:<45}"
            f"{count}"
        )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    print()
    print(
        "Output:"
    )

    print(
        f"  {INTEGRATION_OUTPUT_PATH}"
    )

    print()
    print(
        "KNOWLEDGE BASE → GRAPH INTEGRATION SELF-TEST: PASSED"
    )
    print("=" * 78)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    _self_test()