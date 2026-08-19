"""
HealthLens Recommendation Integration Test
===========================================

End-to-end pipeline:

    Knowledge Graph
        ↓
    Knowledge Base Registry
        ↓
    Contextual Reasoner
        ↓
    Intervention Prioritizer
        ↓
    Recommendation Engine
        ↓
    RecommendationResult
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ============================================================================
# PROJECT PATHS
# ============================================================================

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
    / "recommendations"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "healthlens_recommendations.json"
)


# ============================================================================
# TEST MEMBER
# ============================================================================

TEST_MEMBER_UUID = (
    "1e7909f8-39b2-3c7e-1fda-a6c3256dc061"
)


def canonical_member_id(
    member_id: str,
) -> str:
    """
    Convert source UUID into the canonical graph member ID.
    """

    value = str(
        member_id
    ).strip()

    if value.startswith(
        "member:"
    ):
        return value

    return (
        "member:"
        + value.replace(
            "-",
            "_",
        )
    )


TEST_MEMBER_ID = canonical_member_id(
    TEST_MEMBER_UUID
)


# ============================================================================
# IMPORTS
# ============================================================================

from src.knowledge_base.registry import (
    build_registry,
)

from src.reasoning.contextual_reasoner import (
    ContextualReasoner,
    load_graph,
)

from src.prioritization.intervention_prioritizer import (
    InterventionPrioritizer,
)

from src.recommendations.recommendation_engine import (
    RecommendationEngine,
    RecommendationResult,
    serialize_recommendations,
    validate_recommendation_result,
)


# ============================================================================
# HELPERS
# ============================================================================

def _get_attr(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """
    Retrieve an attribute from either an object or dictionary.
    """

    if obj is None:
        return default

    if isinstance(
        obj,
        dict,
    ):

        for name in names:

            if name in obj:

                value = obj[name]

                if value is not None:
                    return value

        return default

    for name in names:

        if hasattr(
            obj,
            name,
        ):

            value = getattr(
                obj,
                name,
            )

            if value is not None:
                return value

    return default


def _collection(
    obj: Any,
    *names: str,
) -> list[Any]:
    """
    Safely retrieve a collection.
    """

    value = _get_attr(
        obj,
        *names,
        default=[],
    )

    if value is None:
        return []

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return list(value)

    return [value]


def _pass(
    label: str,
) -> None:
    print(
        f"{label:<45} PASS"
    )


# ============================================================================
# GRAPH VALIDATION
# ============================================================================

def _validate_graph(
    graph: Any,
) -> None:
    """
    Validate that the graph contains nodes and the
    canonical test member.
    """

    if graph is None:

        raise AssertionError(
            "Knowledge Graph is None."
        )

    if not isinstance(
        graph,
        dict,
    ):

        raise AssertionError(
            "Knowledge Graph must be a dictionary."
        )

    nodes = graph.get(
        "nodes",
        [],
    )

    if not nodes:

        raise AssertionError(
            "Knowledge Graph contains no nodes."
        )

    # ------------------------------------------------------------------------
    # Member lookup
    # ------------------------------------------------------------------------

    found = False

    for node in nodes:

        if not isinstance(
            node,
            dict,
        ):
            continue

        possible_ids = {
            str(
                node.get(
                    "id",
                    "",
                )
            ),
            str(
                node.get(
                    "node_id",
                    "",
                )
            ),
            str(
                node.get(
                    "member_id",
                    "",
                )
            ),
            str(
                node.get(
                    "uuid",
                    "",
                )
            ),
        }

        if (
            TEST_MEMBER_ID
            in possible_ids
            or TEST_MEMBER_UUID
            in possible_ids
        ):

            found = True
            break

    if not found:

        raise AssertionError(
            "Test member was not found in graph.\n"
            f"Expected: {TEST_MEMBER_ID}"
        )


# ============================================================================
# REASONING VALIDATION
# ============================================================================

def _validate_reasoning(
    reasoning_result: Any,
) -> None:
    """
    Validate the real reasoning result.
    """

    if reasoning_result is None:

        raise AssertionError(
            "Reasoning result is None."
        )

    member_id = _get_attr(
        reasoning_result,
        "member_id",
        default=None,
    )

    if not member_id:

        raise AssertionError(
            "Reasoning result has no member ID."
        )

    risk_probability = _get_attr(
        reasoning_result,
        "risk_probability",
        "risk_score",
        default=None,
    )

    if risk_probability is None:

        raise AssertionError(
            "Reasoning result has no risk probability."
        )

    risk_probability = float(
        risk_probability
    )

    if not (
        0.0
        <= risk_probability
        <= 1.0
    ):

        raise AssertionError(
            "Risk probability must be between 0 and 1."
        )

    sdoh_factors = _collection(
        reasoning_result,
        "sdoh_factors",
        "relevant_sdoh_factors",
    )

    if not sdoh_factors:

        raise AssertionError(
            "Reasoning produced no SDOH factors."
        )

    domains = _collection(
        reasoning_result,
        "sdoh_domains",
        "domains",
    )

    if not domains:

        raise AssertionError(
            "Reasoning produced no SDOH domains."
        )

    evidence = _collection(
        reasoning_result,
        "evidence_records",
        "evidence",
    )

    if not evidence:

        raise AssertionError(
            "Reasoning produced no evidence."
        )

    candidates = _collection(
        reasoning_result,
        "intervention_candidates",
        "candidates",
    )

    if not candidates:

        raise AssertionError(
            "Reasoning produced no intervention candidates."
        )


# ============================================================================
# PRIORITIZATION VALIDATION
# ============================================================================

def _validate_prioritization(
    prioritization_result: Any,
) -> list[Any]:
    """
    Validate the real prioritization result.
    """

    if prioritization_result is None:

        raise AssertionError(
            "Prioritization result is None."
        )

    priorities = _collection(
        prioritization_result,
        "priorities",
        "ranked_interventions",
        "interventions",
    )

    if not priorities:

        raise AssertionError(
            "Prioritizer returned no interventions."
        )

    previous_score = None

    for priority in priorities:

        score = _get_attr(
            priority,
            "priority_score",
            "score",
            default=None,
        )

        if score is None:

            raise AssertionError(
                "Prioritized intervention has no score."
            )

        score = float(
            score
        )

        if score < 0:

            raise AssertionError(
                "Prioritized intervention has negative score."
            )

        if (
            previous_score is not None
            and score > previous_score
        ):

            raise AssertionError(
                "Prioritized interventions are not "
                "sorted by descending score."
            )

        previous_score = score

    return priorities


# ============================================================================
# RECOMMENDATION VALIDATION
# ============================================================================

def _validate_recommendations(
    result: RecommendationResult,
    priorities: list[Any],
) -> None:
    """
    Validate recommendation output against the
    prioritization output.
    """

    validate_recommendation_result(
        result
    )

    recommendations = (
        result.recommendations
    )

    if not recommendations:

        raise AssertionError(
            "Recommendation engine produced no recommendations."
        )

    if len(
        recommendations
    ) != len(
        priorities
    ):

        raise AssertionError(
            "Recommendation count does not match "
            "prioritized intervention count."
        )

    # ------------------------------------------------------------------------
    # Intervention propagation
    # ------------------------------------------------------------------------

    priority_ids = {
        str(
            _get_attr(
                item,
                "intervention_id",
                "id",
                default="",
            )
        )
        for item in priorities
    }

    recommendation_ids = {
        str(
            item.intervention_id
        )
        for item in recommendations
    }

    if priority_ids != recommendation_ids:

        raise AssertionError(
            "Intervention IDs were not propagated "
            "correctly into recommendations."
        )

    # ------------------------------------------------------------------------
    # Score propagation
    # ------------------------------------------------------------------------

    priority_scores = {
        str(
            _get_attr(
                item,
                "intervention_id",
                "id",
                default="",
            )
        ):
        float(
            _get_attr(
                item,
                "priority_score",
                "score",
                default=0,
            )
        )
        for item in priorities
    }

    for recommendation in recommendations:

        expected_score = priority_scores.get(
            recommendation.intervention_id
        )

        if expected_score is None:

            raise AssertionError(
                "Recommendation references an unknown "
                "intervention."
            )

        if (
            float(
                recommendation.priority_score
            )
            != expected_score
        ):

            raise AssertionError(
                "Priority score was not propagated correctly."
            )

    # ------------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------------

    for recommendation in recommendations:

        if not recommendation.action:

            raise AssertionError(
                "Recommendation has no action."
            )

        if not recommendation.rationale:

            raise AssertionError(
                "Recommendation has no rationale."
            )

        if not recommendation.matched_factors:

            raise AssertionError(
                "Recommendation has no matched SDOH factors."
            )

        if recommendation.evidence_count <= 0:

            raise AssertionError(
                "Recommendation has no supporting evidence."
            )


# ============================================================================
# MAIN INTEGRATION TEST
# ============================================================================

def run_integration_test() -> None:

    print("=" * 70)

    print(
        "HEALTHLENS PRIORITIZATION → RECOMMENDATION "
        "INTEGRATION"
    )

    print("=" * 70)

    # ========================================================================
    # 1. KNOWLEDGE GRAPH
    # ========================================================================

    if not GRAPH_PATH.exists():

        raise FileNotFoundError(
            "Knowledge Graph not found:\n"
            f"{GRAPH_PATH}"
        )

    graph = load_graph(
        GRAPH_PATH
    )

    _validate_graph(
        graph
    )

    _pass(
        "Knowledge Graph loading:"
    )

    nodes = graph.get(
        "nodes",
        [],
    )

    print(
        f"{'Graph nodes:':<45}"
        f"{len(nodes)}"
    )

    # ========================================================================
    # 2. REGISTRY
    # ========================================================================

    registry = build_registry()

    _pass(
        "Knowledge Base Registry:"
    )

    # ========================================================================
    # 3. CONTEXTUAL REASONER
    # ========================================================================

    reasoner = ContextualReasoner(
        graph=graph,
        registry=registry,
    )

    _pass(
        "Contextual Reasoner:"
    )

    # ========================================================================
    # 4. REASONING
    # ========================================================================

    print(
        f"{'Test member:':<45}"
        f"{TEST_MEMBER_ID}"
    )

    reasoning_result = reasoner.reason(
        TEST_MEMBER_ID
    )

    _validate_reasoning(
        reasoning_result
    )

    _pass(
        "Member reasoning:"
    )

    # ========================================================================
    # 5. REASONING SUMMARY
    # ========================================================================

    risk_probability = _get_attr(
        reasoning_result,
        "risk_probability",
        "risk_score",
        default=None,
    )

    risk_band = _get_attr(
        reasoning_result,
        "risk_band",
        "risk_level",
        default=None,
    )

    sdoh_factors = _collection(
        reasoning_result,
        "sdoh_factors",
        "relevant_sdoh_factors",
    )

    domains = _collection(
        reasoning_result,
        "sdoh_domains",
        "domains",
    )

    clinical_factors = _collection(
        reasoning_result,
        "clinical_factors",
        "clinical_context",
    )

    evidence = _collection(
        reasoning_result,
        "evidence_records",
        "evidence",
    )

    candidates = _collection(
        reasoning_result,
        "intervention_candidates",
        "candidates",
    )

    print(
        f"{'Risk probability:':<45}"
        f"{risk_probability}"
    )

    print(
        f"{'Risk band:':<45}"
        f"{risk_band}"
    )

    print(
        f"{'SDOH factors:':<45}"
        f"{len(sdoh_factors)}"
    )

    print(
        f"{'SDOH domains:':<45}"
        f"{len(domains)}"
    )

    print(
        f"{'Clinical factors:':<45}"
        f"{len(clinical_factors)}"
    )

    print(
        f"{'Evidence records:':<45}"
        f"{len(evidence)}"
    )

    print(
        f"{'Intervention candidates:':<45}"
        f"{len(candidates)}"
    )

    # ========================================================================
    # 6. PRIORITIZER
    # ========================================================================

    prioritizer = (
        InterventionPrioritizer()
    )

    _pass(
        "Intervention Prioritizer:"
    )

    # ========================================================================
    # 7. PRIORITIZATION
    # ========================================================================

    prioritization_result = (
        prioritizer.prioritize(
            reasoning_result
        )
    )

    priorities = (
        _validate_prioritization(
            prioritization_result
        )
    )

    _pass(
        "Intervention prioritization:"
    )

    # ========================================================================
    # 8. RECOMMENDATION ENGINE
    # ========================================================================

    recommendation_engine = (
        RecommendationEngine()
    )

    _pass(
        "Recommendation Engine:"
    )

    # ========================================================================
    # 9. RECOMMENDATIONS
    # ========================================================================

    recommendation_result = (
        recommendation_engine.generate(
            reasoning_result,
            prioritization_result,
        )
    )

    _validate_recommendations(
        recommendation_result,
        priorities,
    )

    _pass(
        "Recommendation generation:"
    )

    # ========================================================================
    # 10. RISK PROPAGATION
    # ========================================================================

    if (
        recommendation_result.risk_probability
        != float(
            risk_probability
        )
    ):

        raise AssertionError(
            "Risk probability changed between "
            "reasoning and recommendation layers."
        )

    if (
        recommendation_result.risk_band
        != str(
            risk_band
        )
    ):

        raise AssertionError(
            "Risk band changed between reasoning "
            "and recommendation layers."
        )

    _pass(
        "Risk propagation:"
    )

    # ========================================================================
    # 11. MEMBER PROPAGATION
    # ========================================================================

    if (
        recommendation_result.member_id
        != TEST_MEMBER_ID
    ):

        raise AssertionError(
            "Member ID was not propagated correctly."
        )

    _pass(
        "Member ID propagation:"
    )

    # ========================================================================
    # 12. SERIALIZATION
    # ========================================================================

    serialized = serialize_recommendations(
        recommendation_result
    )

    if not isinstance(
        serialized,
        dict,
    ):

        raise AssertionError(
            "Recommendation serialization failed."
        )

    if (
        "recommendations"
        not in serialized
    ):

        raise AssertionError(
            "Serialized result contains no recommendations."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            serialized,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    _pass(
        "Recommendation serialization:"
    )

    # ========================================================================
    # 13. FINAL OUTPUT
    # ========================================================================

    print()

    print("=" * 70)

    print(
        "RECOMMENDATION RESULT"
    )

    print("=" * 70)

    print(
        f"Member ID:                 "
        f"{recommendation_result.member_id}"
    )

    print(
        f"Risk probability:          "
        f"{recommendation_result.risk_probability}"
    )

    print(
        f"Risk band:                 "
        f"{recommendation_result.risk_band}"
    )

    print(
        f"Recommendations:           "
        f"{recommendation_result.total_recommendations}"
    )

    print()

    print(
        "ACTIONABLE RECOMMENDATIONS"
    )

    print("-" * 70)

    for recommendation in (
        recommendation_result.recommendations
    ):

        print(
            f"{recommendation.rank}. "
            f"{recommendation.intervention_id} | "
            f"{recommendation.intervention_name}"
        )

        print(
            f"   Domain:              "
            f"{recommendation.domain}"
        )

        print(
            f"   Priority score:      "
            f"{recommendation.priority_score}"
        )

        print(
            f"   Priority band:       "
            f"{recommendation.priority_band}"
        )

        print(
            f"   Urgency:             "
            f"{recommendation.urgency}"
        )

        print(
            f"   Matched factors:     "
            f"{len(recommendation.matched_factors)}"
        )

        print(
            f"   Evidence:            "
            f"{recommendation.evidence_count}"
        )

        print(
            f"   Action:              "
            f"{recommendation.action}"
        )

        print(
            f"   Rationale:           "
            f"{recommendation.rationale}"
        )

        print()

    print(
        "Output:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print()

    print("=" * 70)

    print(
        "PRIORITIZATION → RECOMMENDATION "
        "INTEGRATION SELF-TEST: PASSED"
    )

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_integration_test()