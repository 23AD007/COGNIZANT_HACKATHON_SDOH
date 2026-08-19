"""
HealthLens
==========

Member Risk -> Contextual Reasoning Integration

Production flow:

    Member Risk Model
          |
          v
    Member Risk Score Adapter
          |
          v
    Canonical Member ID
          |
          v
    Knowledge Graph
          |
          v
    Contextual Reasoner
          |
          v
    Integrated Reasoning Result

Important
---------
This module:

- uses the existing trained member-risk model output
- uses the existing Knowledge Graph
- uses the existing Knowledge Base Registry
- does NOT retrain the risk model
- does NOT invent SDOH features
- does NOT create synthetic graph relationships
- does NOT modify the Knowledge Graph
- normalizes graph IDs only for lookup
- fails loudly if the persisted graph has no relationships
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_SCORE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member_risk_model_predictions.csv"
)

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
    / "reasoning"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "member_risk_reasoning_integration.json"
)


# ============================================================================
# TEST MEMBER
# ============================================================================

TEST_MEMBER_ID = (
    "1e7909f8-39b2-3c7e-1fda-a6c3256dc061"
)

EXPECTED_GRAPH_NODES = 2246
EXPECTED_GRAPH_RELATIONSHIPS = 3407


# ============================================================================
# IMPORTS
# ============================================================================

from src.modeling.member_risk import (
    MemberRiskScoreAdapter,
)

from src.knowledge_base.registry import (
    build_registry,
)

from src.reasoning.contextual_reasoner import (
    ContextualReasoner,
)


# ============================================================================
# PRINT HELPERS
# ============================================================================


def _pass(label: str) -> None:

    print(
        f"{label:<42} PASS"
    )


# ============================================================================
# GENERIC HELPERS
# ============================================================================


def _as_dict(value: Any) -> dict[str, Any]:

    if isinstance(value, Mapping):
        return dict(value)

    if hasattr(value, "to_dict"):

        result = value.to_dict()

        if isinstance(result, Mapping):
            return dict(result)

    if hasattr(value, "__dict__"):
        return dict(vars(value))

    raise TypeError(
        f"Cannot serialize object of type {type(value)!r}"
    )


def _serialize(value: Any) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, Mapping):

        return {
            str(k): _serialize(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            _serialize(item)
            for item in value
        ]

    if hasattr(value, "to_dict"):

        return _serialize(
            value.to_dict()
        )

    if hasattr(value, "__dict__"):

        return _serialize(
            vars(value)
        )

    return str(value)


# ============================================================================
# MEMBER ID NORMALIZATION
# ============================================================================


def canonical_graph_member_id(
    member_id: str,
) -> str:

    """
    Convert the Synthea UUID into the canonical graph MEMBER id.

    Example:

        1e7909f8-39b2-3c7e-1fda-a6c3256dc061

    becomes:

        member:1e7909f8_39b2_3c7e_1fda_a6c3256dc061
    """

    value = str(
        member_id
    ).strip()

    if value.startswith("member:"):

        suffix = value[
            len("member:"):
        ]

    else:

        suffix = value

    suffix = suffix.replace(
        "-",
        "_",
    )

    return (
        "member:"
        + suffix
    )


def equivalent_member_ids(
    member_id: str,
) -> set[str]:

    """
    IDs accepted when locating a member.

    No new member is created.
    These are only lookup aliases.
    """

    raw = str(
        member_id
    ).strip()

    canonical = (
        canonical_graph_member_id(
            raw
        )
    )

    uuid_form = (
        canonical[
            len("member:"):
        ]
        .replace(
            "_",
            "-",
        )
    )

    return {
        raw,
        canonical,
        uuid_form,
        f"member:{uuid_form}",
    }


# ============================================================================
# GRAPH EXTRACTION
# ============================================================================


def _extract_nodes(
    graph: Mapping[str, Any],
) -> list[dict[str, Any]]:

    nodes = graph.get(
        "nodes"
    )

    if isinstance(
        nodes,
        list,
    ):

        return [
            dict(node)
            for node in nodes
            if isinstance(
                node,
                Mapping,
            )
        ]

    if isinstance(
        nodes,
        Mapping,
    ):

        return [
            dict(node)
            for node in nodes.values()
            if isinstance(
                node,
                Mapping,
            )
        ]

    return []


def _extract_relationships(
    graph: Mapping[str, Any],
) -> list[dict[str, Any]]:

    """
    Read the actual relationship collection from the persisted graph.

    Supported persisted keys:

        relationships
        edges

    Also handles mapping-style collections.
    """

    relationships = graph.get(
        "relationships"
    )

    if relationships is None:

        relationships = graph.get(
            "edges"
        )

    if relationships is None:

        return []

    if isinstance(
        relationships,
        list,
    ):

        return [
            dict(rel)
            for rel in relationships
            if isinstance(
                rel,
                Mapping,
            )
        ]

    if isinstance(
        relationships,
        Mapping,
    ):

        return [
            dict(rel)
            for rel in relationships.values()
            if isinstance(
                rel,
                Mapping,
            )
        ]

    return []


# ============================================================================
# RELATIONSHIP FIELD EXTRACTION
# ============================================================================


def _relationship_source(
    relationship: Mapping[str, Any],
) -> str:

    for key in (
        "source",
        "from",
        "source_id",
        "from_id",
        "src",
    ):

        value = relationship.get(
            key
        )

        if value is not None:

            return str(
                value
            )

    return ""


def _relationship_target(
    relationship: Mapping[str, Any],
) -> str:

    for key in (
        "target",
        "to",
        "target_id",
        "to_id",
        "dst",
    ):

        value = relationship.get(
            key
        )

        if value is not None:

            return str(
                value
            )

    return ""


def _relationship_type(
    relationship: Mapping[str, Any],
) -> str:

    for key in (
        "type",
        "relationship_type",
        "label",
    ):

        value = relationship.get(
            key
        )

        if value is not None:

            return str(
                value
            )

    return ""


# ============================================================================
# NODE FIELD EXTRACTION
# ============================================================================


def _node_id(
    node: Mapping[str, Any],
) -> str:

    for key in (
        "id",
        "node_id",
        "key",
    ):

        value = node.get(
            key
        )

        if value is not None:

            return str(
                value
            )

    return ""


def _node_type(
    node: Mapping[str, Any],
) -> str:

    for key in (
        "type",
        "node_type",
        "label",
    ):

        value = node.get(
            key
        )

        if value is not None:

            return str(
                value
            )

    return ""


# ============================================================================
# GRAPH NORMALIZATION
# ============================================================================


def normalize_graph(
    raw_graph: Mapping[str, Any],
) -> dict[str, Any]:

    """
    Normalize only representation.

    This function NEVER invents relationships.

    It copies the persisted nodes and relationships into the exact
    top-level collections expected by the reasoning layer.
    """

    nodes = _extract_nodes(
        raw_graph
    )

    relationships = _extract_relationships(
        raw_graph
    )

    normalized = dict(
        raw_graph
    )

    normalized[
        "nodes"
    ] = nodes

    normalized[
        "relationships"
    ] = relationships

    return normalized


# ============================================================================
# GRAPH LOADING
# ============================================================================


def load_integration_graph() -> dict[str, Any]:

    if not GRAPH_PATH.exists():

        raise FileNotFoundError(
            "Knowledge Graph not found:\n"
            f"{GRAPH_PATH}"
        )

    with GRAPH_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:

        raw_graph = json.load(
            handle
        )

    if not isinstance(
        raw_graph,
        Mapping,
    ):

        raise ValueError(
            "Knowledge Graph JSON must contain "
            "an object at the root."
        )

    graph = normalize_graph(
        raw_graph
    )

    nodes = _extract_nodes(
        graph
    )

    relationships = _extract_relationships(
        graph
    )

    if not nodes:

        raise AssertionError(
            "Knowledge Graph loaded but contains "
            "zero nodes.\n\n"
            f"Graph path:\n{GRAPH_PATH}"
        )

    if not relationships:

        # IMPORTANT:
        # Do not fabricate relationships.
        # This means the persisted graph artifact is not
        # the graph expected by the reasoning layer.
        raise AssertionError(
            "Knowledge Graph loaded but contains "
            "zero relationships.\n\n"
            "The reasoning layer requires the original "
            "member-level graph relationships.\n\n"
            f"Graph path:\n{GRAPH_PATH}\n\n"
            "Expected graph:\n"
            f"  nodes: {EXPECTED_GRAPH_NODES}\n"
            f"  relationships: "
            f"{EXPECTED_GRAPH_RELATIONSHIPS}"
        )

    return graph


# ============================================================================
# GRAPH VALIDATION
# ============================================================================


def find_member_node(
    nodes: list[dict[str, Any]],
    member_id: str,
) -> dict[str, Any] | None:

    accepted = (
        equivalent_member_ids(
            member_id
        )
    )

    for node in nodes:

        node_id = _node_id(
            node
        )

        node_type = _node_type(
            node
        ).upper()

        if node_type != "MEMBER":
            continue

        if node_id in accepted:
            return node

        normalized = (
            node_id
            .strip()
        )

        if normalized in accepted:
            return node

    return None


def validate_graph(
    graph: Mapping[str, Any],
) -> dict[str, Any]:

    nodes = _extract_nodes(
        graph
    )

    relationships = _extract_relationships(
        graph
    )

    if len(nodes) != EXPECTED_GRAPH_NODES:

        raise AssertionError(
            "Unexpected Knowledge Graph node count.\n"
            f"Expected: {EXPECTED_GRAPH_NODES}\n"
            f"Actual:   {len(nodes)}"
        )

    if len(relationships) != EXPECTED_GRAPH_RELATIONSHIPS:

        raise AssertionError(
            "Unexpected Knowledge Graph relationship count.\n"
            f"Expected: {EXPECTED_GRAPH_RELATIONSHIPS}\n"
            f"Actual:   {len(relationships)}"
        )

    member = find_member_node(
        nodes,
        TEST_MEMBER_ID,
    )

    if member is None:

        raise AssertionError(
            "Test member was not found in Knowledge Graph.\n"
            f"Risk-model member ID: {TEST_MEMBER_ID}\n"
            f"Expected graph ID: "
            f"{canonical_graph_member_id(TEST_MEMBER_ID)}"
        )

    return member


# ============================================================================
# MEMBER RISK
# ============================================================================


def load_member_risk_adapter() -> MemberRiskScoreAdapter:

    adapter = (
        MemberRiskScoreAdapter()
    )

    return adapter


# ============================================================================
# RISK LOOKUP
# ============================================================================


def get_member_risk(
    adapter: MemberRiskScoreAdapter,
    member_id: str,
) -> Any:

    """
    Use the existing risk-score adapter.

    The adapter is responsible for reading:

        data/processed/member_risk_model_predictions.csv

    and returning the trained model's stored prediction.
    """

    getter_names = (
        "get_member_risk",
        "get_risk_score",
        "get_member",
        "lookup",
    )

    for name in getter_names:

        getter = getattr(
            adapter,
            name,
            None,
        )

        if not callable(
            getter
        ):
            continue

        try:

            result = getter(
                member_id
            )

        except (
            KeyError,
            ValueError,
        ):

            continue

        if result is not None:

            return result

    raise AttributeError(
        "MemberRiskScoreAdapter does not expose "
        "a supported member-risk lookup method."
    )


# ============================================================================
# RISK FIELD EXTRACTION
# ============================================================================


def extract_risk_probability(
    risk_record: Any,
) -> float:

    data = _as_dict(
        risk_record
    )

    candidates = (
        "risk_probability",
        "probability",
        "predicted_probability",
        "risk_score",
    )

    for key in candidates:

        if key in data:

            value = data[
                key
            ]

            if value is not None:

                probability = float(
                    value
                )

                if not (
                    0.0
                    <= probability
                    <= 1.0
                ):

                    raise ValueError(
                        "Risk probability is outside "
                        "[0, 1]."
                    )

                return probability

    raise KeyError(
        "Risk record does not contain a "
        "risk probability."
    )


def extract_risk_band(
    risk_record: Any,
) -> str:

    data = _as_dict(
        risk_record
    )

    for key in (
        "risk_band",
        "risk_category",
        "band",
    ):

        value = data.get(
            key
        )

        if value is not None:

            return str(
                value
            )

    raise KeyError(
        "Risk record does not contain risk_band."
    )


# ============================================================================
# REASONING RISK PROPAGATION
# ============================================================================


def inject_risk_into_member_node(
    graph: dict[str, Any],
    graph_member_id: str,
    risk_record: Any,
) -> dict[str, Any]:

    """
    Propagate the trained risk result into the in-memory member node.

    This does NOT write to the persisted Knowledge Graph.

    It is required because the Knowledge Graph contains contextual
    information while the trained risk model owns the risk prediction.
    """

    risk_probability = (
        extract_risk_probability(
            risk_record
        )
    )

    risk_band = (
        extract_risk_band(
            risk_record
        )
    )

    nodes = graph[
        "nodes"
    ]

    for node in nodes:

        if (
            _node_id(node)
            == graph_member_id
        ):

            properties = node.get(
                "properties"
            )

            if not isinstance(
                properties,
                dict,
            ):

                properties = {}

                node[
                    "properties"
                ] = properties

            properties[
                "risk_probability"
            ] = risk_probability

            properties[
                "risk_band"
            ] = risk_band

            # Also expose top-level values because the reasoner
            # supports both forms.
            node[
                "risk_probability"
            ] = risk_probability

            node[
                "risk_band"
            ] = risk_band

            return graph

    raise ValueError(
        "Graph member disappeared during "
        "risk propagation: "
        f"{graph_member_id}"
    )


# ============================================================================
# REASONING
# ============================================================================


def run_reasoning(
    graph: dict[str, Any],
    member_id: str,
) -> Any:

    registry = (
        build_registry()
    )

    reasoner = ContextualReasoner(
        graph=graph,
        registry=registry,
    )

    result = reasoner.reason(
        member_id
    )

    return result


# ============================================================================
# SERIALIZATION
# ============================================================================


def serialize_reasoning_result(
    result: Any,
) -> dict[str, Any]:

    return _serialize(
        result
    )


# ============================================================================
# INTEGRATION
# ============================================================================


def integrate_member(
    member_id: str,
) -> dict[str, Any]:

    # ------------------------------------------------------------------------
    # RISK
    # ------------------------------------------------------------------------

    adapter = (
        load_member_risk_adapter()
    )

    risk_record = (
        get_member_risk(
            adapter,
            member_id,
        )
    )

    risk_probability = (
        extract_risk_probability(
            risk_record
        )
    )

    risk_band = (
        extract_risk_band(
            risk_record
        )
    )

    # ------------------------------------------------------------------------
    # GRAPH
    # ------------------------------------------------------------------------

    graph = (
        load_integration_graph()
    )

    member_node = (
        find_member_node(
            graph["nodes"],
            member_id,
        )
    )

    if member_node is None:

        raise ValueError(
            "Member not found in graph.\n"
            f"Risk member ID: {member_id}\n"
            f"Canonical graph ID: "
            f"{canonical_graph_member_id(member_id)}"
        )

    graph_member_id = (
        _node_id(
            member_node
        )
    )

    # ------------------------------------------------------------------------
    # RISK -> GRAPH CONTEXT
    # ------------------------------------------------------------------------

    graph = (
        inject_risk_into_member_node(
            graph,
            graph_member_id,
            risk_record,
        )
    )

    # ------------------------------------------------------------------------
    # REASONING
    # ------------------------------------------------------------------------

    reasoning_result = (
        run_reasoning(
            graph,
            graph_member_id,
        )
    )

    serialized_reasoning = (
        serialize_reasoning_result(
            reasoning_result
        )
    )

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    if (
        serialized_reasoning.get(
            "risk_probability"
        )
        is None
    ):

        raise AssertionError(
            "Risk probability was not propagated "
            "into reasoning result."
        )

    if (
        abs(
            float(
                serialized_reasoning[
                    "risk_probability"
                ]
            )
            - risk_probability
        )
        > 1e-12
    ):

        raise AssertionError(
            "Risk probability changed during "
            "risk -> reasoning integration."
        )

    if (
        str(
            serialized_reasoning.get(
                "risk_band"
            )
        )
        != risk_band
    ):

        raise AssertionError(
            "Risk band changed during "
            "risk -> reasoning integration."
        )

    return {
        "member_id": member_id,
        "graph_member_id": graph_member_id,
        "risk_probability": risk_probability,
        "risk_band": risk_band,
        "reasoning": serialized_reasoning,
        "graph_counts": {
            "nodes": len(
                graph["nodes"]
            ),
            "relationships": len(
                graph["relationships"]
            ),
        },
    }


# ============================================================================
# SELF TEST
# ============================================================================


def run_integration_test() -> None:

    print(
        "=" * 78
    )

    print(
        "HEALTHLENS MEMBER RISK → CONTEXTUAL REASONING"
    )

    print(
        "=" * 78
    )

    # ------------------------------------------------------------------------
    # RISK ADAPTER
    # ------------------------------------------------------------------------

    adapter = (
        load_member_risk_adapter()
    )

    print(
        "Member risk scores:                      PASS"
    )

    risk_record = (
        get_member_risk(
            adapter,
            TEST_MEMBER_ID,
        )
    )

    print(
        "Risk record lookup:                      PASS"
    )

    probability = (
        extract_risk_probability(
            risk_record
        )
    )

    band = (
        extract_risk_band(
            risk_record
        )
    )

    print(
        f"Test member:                             "
        f"{TEST_MEMBER_ID}"
    )

    print(
        "Member ID lookup:                        PASS"
    )

    print(
        f"Risk probability:                       "
        f"{probability:.6f}"
    )

    print(
        f"Risk band:                              "
        f"{band}"
    )

    # ------------------------------------------------------------------------
    # GRAPH
    # ------------------------------------------------------------------------

    graph = (
        load_integration_graph()
    )

    nodes = (
        graph["nodes"]
    )

    relationships = (
        graph["relationships"]
    )

    print(
        "Knowledge Graph loading:                 PASS"
    )

    print(
        f"Graph nodes:                             "
        f"{len(nodes)}"
    )

    print(
        f"Graph relationships:                     "
        f"{len(relationships)}"
    )

    # ------------------------------------------------------------------------
    # GRAPH STRUCTURE
    # ------------------------------------------------------------------------

    member_node = (
        validate_graph(
            graph
        )
    )

    graph_member_id = (
        _node_id(
            member_node
        )
    )

    print(
        f"Graph member ID:                         "
        f"{graph_member_id}"
    )

    print(
        "Graph member resolution:                 PASS"
    )

    expected_id = (
        canonical_graph_member_id(
            TEST_MEMBER_ID
        )
    )

    if graph_member_id != expected_id:

        raise AssertionError(
            "Canonical graph member ID mismatch.\n"
            f"Expected: {expected_id}\n"
            f"Actual:   {graph_member_id}"
        )

    print(
        "Canonical member ID resolution:          PASS"
    )

    # ------------------------------------------------------------------------
    # RISK PROPAGATION
    # ------------------------------------------------------------------------

    graph = (
        inject_risk_into_member_node(
            graph,
            graph_member_id,
            risk_record,
        )
    )

    print(
        "Risk propagation:                        PASS"
    )

    # ------------------------------------------------------------------------
    # KNOWLEDGE BASE
    # ------------------------------------------------------------------------

    registry = (
        build_registry()
    )

    print(
        "Knowledge Base Registry:                 PASS"
    )

    # ------------------------------------------------------------------------
    # REASONER
    # ------------------------------------------------------------------------

    reasoner = ContextualReasoner(
        graph=graph,
        registry=registry,
    )

    print(
        "Contextual Reasoner construction:        PASS"
    )

    # ------------------------------------------------------------------------
    # REASONING
    # ------------------------------------------------------------------------

    result = (
        reasoner.reason(
            graph_member_id
        )
    )

    print(
        "Contextual reasoning:                    PASS"
    )

    serialized = (
        serialize_reasoning_result(
            result
        )
    )

    # ------------------------------------------------------------------------
    # VALIDATE RESULT
    # ------------------------------------------------------------------------

    result_member_id = str(
        serialized.get(
            "member_id",
            "",
        )
    )

    if result_member_id != graph_member_id:

        raise AssertionError(
            "Member ID was not propagated correctly.\n"
            f"Expected: {graph_member_id}\n"
            f"Actual:   {result_member_id}"
        )

    print(
        "Member ID propagation:                   PASS"
    )

    result_probability = serialized.get(
        "risk_probability"
    )

    if result_probability is None:

        raise AssertionError(
            "Reasoning result has no risk probability."
        )

    if abs(
        float(
            result_probability
        )
        - probability
    ) > 1e-12:

        raise AssertionError(
            "Risk probability changed between "
            "risk model and reasoning layer."
        )

    print(
        "Risk probability propagation:            PASS"
    )

    result_band = serialized.get(
        "risk_band"
    )

    if str(
        result_band
    ) != band:

        raise AssertionError(
            "Risk band changed between "
            "risk model and reasoning layer."
        )

    print(
        "Risk band propagation:                   PASS"
    )

    sdoh_factors = (
        serialized.get(
            "sdoh_factors",
            [],
        )
        or []
    )

    sdoh_domains = (
        serialized.get(
            "sdoh_domains",
            [],
        )
        or []
    )

    clinical_factors = (
        serialized.get(
            "clinical_factors",
            [],
        )
        or []
    )

    evidence_records = (
        serialized.get(
            "evidence_records",
            [],
        )
        or []
    )

    intervention_candidates = (
        serialized.get(
            "intervention_candidates",
            [],
        )
        or []
    )

    if not sdoh_factors:

        raise AssertionError(
            "No SDOH factors were produced."
        )

    print(
        "SDOH context propagation:                PASS"
    )

    if not sdoh_domains:

        raise AssertionError(
            "No SDOH domains were produced."
        )

    print(
        "SDOH domain propagation:                 PASS"
    )

    if not clinical_factors:

        raise AssertionError(
            "No clinical factors were produced."
        )

    print(
        "Clinical context propagation:            PASS"
    )

    if not evidence_records:

        raise AssertionError(
            "No evidence records were produced."
        )

    print(
        "Evidence propagation:                    PASS"
    )

    if not intervention_candidates:

        raise AssertionError(
            "No intervention candidates were produced."
        )

    print(
        "Intervention candidate propagation:      PASS"
    )

    # ------------------------------------------------------------------------
    # SAVE OUTPUT
    # ------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    integration_output = {
        "member_id": TEST_MEMBER_ID,
        "graph_member_id": graph_member_id,
        "risk_probability": probability,
        "risk_band": band,
        "graph": {
            "nodes": len(
                nodes
            ),
            "relationships": len(
                relationships
            ),
        },
        "reasoning": serialized,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            _serialize(
                integration_output
            ),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "Reasoning serialization:                 PASS"
    )

    print()
    print(
        "=" * 78
    )

    print(
        "INTEGRATION RESULT"
    )

    print(
        "=" * 78
    )

    print(
        f"Member ID:                 "
        f"{TEST_MEMBER_ID}"
    )

    print(
        f"Graph member ID:           "
        f"{graph_member_id}"
    )

    print(
        f"Risk probability:          "
        f"{probability}"
    )

    print(
        f"Risk band:                 "
        f"{band}"
    )

    print(
        f"SDOH factors:              "
        f"{len(sdoh_factors)}"
    )

    print(
        f"SDOH domains:              "
        f"{len(sdoh_domains)}"
    )

    print(
        f"Clinical factors:          "
        f"{len(clinical_factors)}"
    )

    print(
        f"Evidence records:          "
        f"{len(evidence_records)}"
    )

    print(
        f"Intervention candidates:   "
        f"{len(intervention_candidates)}"
    )

    print()
    print(
        "Output:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print()
    print(
        "=" * 78
    )

    print(
        "MEMBER RISK → CONTEXTUAL REASONING "
        "INTEGRATION SELF-TEST: PASSED"
    )

    print(
        "=" * 78
    )


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":

    run_integration_test()