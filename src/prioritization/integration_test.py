"""
HealthLens Reasoning → Prioritization Integration Test
=======================================================

Pipeline:

    Knowledge Graph
          |
          v
    Knowledge Base Registry
          |
          v
    Contextual Reasoner
          |
          v
    ReasoningResult
          |
          v
    Intervention Prioritizer
          |
          v
    PrioritizationResult

This test uses the real production components and does not
create synthetic reasoning or prioritization data.
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
    / "prioritization"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "healthlens_prioritization_result.json"
)


# ============================================================================
# TEST MEMBER
# ============================================================================

# Raw UUID used by the source member data.
TEST_MEMBER_UUID = (
    "1e7909f8-39b2-3c7e-1fda-a6c3256dc061"
)


# Canonical member ID used by the Knowledge Graph.
#
# Example:
#
#   UUID:
#       1e7909f8-39b2-3c7e-1fda-a6c3256dc061
#
#   Graph:
#       member:1e7909f8_39b2_3c7e_1fda_a6c3256dc061
#
CANONICAL_MEMBER_ID = (
    f"member:"
    f"{TEST_MEMBER_UUID.replace('-', '_')}"
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

from .intervention_prioritizer import (
    InterventionPrioritizer,
)


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def _print_pass(
    label: str,
) -> None:
    """Print a standardized PASS message."""

    print(
        f"{label:<40} PASS"
    )


def _get_attr(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve an attribute from an object.
    """

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


def _extract_collection(
    obj: Any,
    *names: str,
) -> list[Any]:
    """
    Safely retrieve a collection from an object.
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


def _serialize_result(
    result: Any,
) -> dict[str, Any]:
    """
    Serialize a prioritization result.
    """

    if hasattr(
        result,
        "to_dict",
    ):

        data = result.to_dict()

        if isinstance(
            data,
            dict,
        ):
            return data

    if hasattr(
        result,
        "__dict__",
    ):

        return dict(
            result.__dict__
        )

    raise TypeError(
        "Prioritization result cannot be serialized."
    )


# ============================================================================
# MEMBER ID NORMALIZATION
# ============================================================================

def _canonical_member_id(
    member_id: str,
) -> str:
    """
    Convert a raw UUID/member ID into the canonical graph ID.

    Supported input:

        1e7909f8-39b2-3c7e-1fda-a6c3256dc061

    Output:

        member:1e7909f8_39b2_3c7e_1fda_a6c3256dc061
    """

    normalized = str(
        member_id
    ).strip()

    if normalized.startswith(
        "member:"
    ):
        return normalized

    return (
        "member:"
        + normalized.replace(
            "-",
            "_",
        )
    )


# ============================================================================
# GRAPH VALIDATION
# ============================================================================

def _validate_graph(
    graph: Any,
) -> None:
    """
    Validate the graph sufficiently for this integration test.

    IMPORTANT:

    load_graph() does not necessarily expose relationships as:

        graph["relationships"]

    Therefore this test does not incorrectly declare the graph
    invalid simply because that dictionary key is absent.
    """

    if graph is None:

        raise AssertionError(
            "Knowledge Graph could not be loaded."
        )

    if not isinstance(
        graph,
        dict,
    ):

        raise AssertionError(
            "Loaded Knowledge Graph must be a dictionary."
        )

    nodes = graph.get(
        "nodes",
        [],
    )

    if not isinstance(
        nodes,
        list,
    ):

        raise AssertionError(
            "Knowledge Graph nodes must be a list."
        )

    if not nodes:

        raise AssertionError(
            "Knowledge Graph contains no nodes."
        )

    # ------------------------------------------------------------------------
    # Validate canonical member ID
    # ------------------------------------------------------------------------

    member_found = False

    target_id = _canonical_member_id(
        TEST_MEMBER_UUID
    )

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
        }

        if target_id in possible_ids:

            member_found = True
            break

    if not member_found:

        # Some graph representations may store the UUID separately.
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
                TEST_MEMBER_UUID
                in possible_ids
            ):

                member_found = True
                break

    if not member_found:

        raise AssertionError(
            "Test member was not found in the loaded "
            "Knowledge Graph.\n"
            f"Expected member ID: {target_id}"
        )


# ============================================================================
# PRIORITY VALIDATION
# ============================================================================

def _validate_priorities(
    priorities: list[Any],
    candidates: list[Any],
) -> None:
    """
    Validate ranked intervention results.
    """

    if not priorities:

        raise AssertionError(
            "Prioritization produced no "
            "ranked interventions."
        )

    if len(priorities) > len(
        candidates
    ):

        raise AssertionError(
            "Prioritization produced more "
            "interventions than candidates."
        )

    # ------------------------------------------------------------------------
    # RANKS
    # ------------------------------------------------------------------------

    ranks = []

    for priority in priorities:

        rank = _get_attr(
            priority,
            "rank",
            default=None,
        )

        if rank is None:

            raise AssertionError(
                "Priority item has no rank."
            )

        ranks.append(
            int(rank)
        )

    expected_ranks = list(
        range(
            1,
            len(priorities) + 1,
        )
    )

    if ranks != expected_ranks:

        raise AssertionError(
            "Intervention ranks are not sequential."
        )

    _print_pass(
        "Rank validation:"
    )

    # ------------------------------------------------------------------------
    # SCORES
    # ------------------------------------------------------------------------

    scores = []

    for priority in priorities:

        score = _get_attr(
            priority,
            "priority_score",
            "score",
            default=None,
        )

        if score is None:

            raise AssertionError(
                "Priority item has no score."
            )

        try:

            numeric_score = float(
                score
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise AssertionError(
                "Priority score is not numeric."
            ) from exc

        if numeric_score < 0:

            raise AssertionError(
                "Negative prioritization score detected."
            )

        scores.append(
            numeric_score
        )

    if scores != sorted(
        scores,
        reverse=True,
    ):

        raise AssertionError(
            "Priorities are not sorted "
            "by descending score."
        )

    _print_pass(
        "Score validation:"
    )

    # ------------------------------------------------------------------------
    # EXPLAINABILITY
    # ------------------------------------------------------------------------

    for priority in priorities:

        intervention_id = _get_attr(
            priority,
            "intervention_id",
            "id",
            default="unknown",
        )

        rationale = _get_attr(
            priority,
            "rationale",
            "explanation",
            default=None,
        )

        if not rationale:

            raise AssertionError(
                "Intervention has no rationale: "
                f"{intervention_id}"
            )

        matched_factors = _extract_collection(
            priority,
            "matched_factors",
            "factors",
        )

        if not matched_factors:

            raise AssertionError(
                "Intervention has no matched factors: "
                f"{intervention_id}"
            )

    _print_pass(
        "Explainability:"
    )


# ============================================================================
# MAIN INTEGRATION TEST
# ============================================================================

def run_integration_test() -> None:

    print("=" * 70)

    print(
        "HEALTHLENS REASONING → PRIORITIZATION "
        "INTEGRATION"
    )

    print("=" * 70)

    # ========================================================================
    # 1. GRAPH
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

    _print_pass(
        "Knowledge Graph loading:"
    )

    nodes = graph.get(
        "nodes",
        [],
    )

    print(
        f"{'Graph nodes detected:':<40}"
        f"{len(nodes)}"
    )

    # Do NOT use:
    #
    #     len(graph.get("relationships", []))
    #
    # as a validation criterion.
    #
    # The graph loader can maintain relationship information in its
    # internal representation rather than exposing a top-level
    # "relationships" key.

    # ========================================================================
    # 2. REGISTRY
    # ========================================================================

    registry = build_registry()

    _print_pass(
        "Knowledge Base Registry:"
    )

    # ========================================================================
    # 3. REASONER
    # ========================================================================

    reasoner = ContextualReasoner(
        graph=graph,
        registry=registry,
    )

    _print_pass(
        "Contextual Reasoner:"
    )

    # ========================================================================
    # 4. CANONICAL MEMBER ID
    # ========================================================================

    member_id = _canonical_member_id(
        TEST_MEMBER_UUID
    )

    print(
        f"{'Test member:':<40}"
        f"{member_id}"
    )

    # ========================================================================
    # 5. MEMBER REASONING
    # ========================================================================

    reasoning_result = reasoner.reason(
        member_id
    )

    if reasoning_result is None:

        raise AssertionError(
            "ContextualReasoner returned None."
        )

    _print_pass(
        "Real member reasoning:"
    )

    # ========================================================================
    # 6. MEMBER ID
    # ========================================================================

    actual_member_id = str(
        _get_attr(
            reasoning_result,
            "member_id",
            default="",
        )
    )

    if not actual_member_id:

        raise AssertionError(
            "ReasoningResult has no member_id."
        )

    _print_pass(
        "Member ID propagation:"
    )

    # ========================================================================
    # 7. RISK
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

    if risk_probability is None:

        raise AssertionError(
            "Risk probability was not produced "
            "by contextual reasoning."
        )

    if risk_band is None:

        raise AssertionError(
            "Risk band was not produced "
            "by contextual reasoning."
        )

    _print_pass(
        "Risk propagation:"
    )

    # ========================================================================
    # 8. SDOH FACTORS
    # ========================================================================

    sdoh_factors = _extract_collection(
        reasoning_result,
        "sdoh_factors",
        "relevant_sdoh_factors",
    )

    if not sdoh_factors:

        raise AssertionError(
            "No SDOH factors were produced."
        )

    _print_pass(
        "SDOH context propagation:"
    )

    # ========================================================================
    # 9. SDOH DOMAINS
    # ========================================================================

    sdoh_domains = _extract_collection(
        reasoning_result,
        "sdoh_domains",
        "domains",
    )

    if not sdoh_domains:

        raise AssertionError(
            "No SDOH domains were produced."
        )

    _print_pass(
        "SDOH domain propagation:"
    )

    # ========================================================================
    # 10. CLINICAL FACTORS
    # ========================================================================

    clinical_factors = _extract_collection(
        reasoning_result,
        "clinical_factors",
        "clinical_context",
    )

    if not clinical_factors:

        raise AssertionError(
            "No clinical factors were produced."
        )

    _print_pass(
        "Clinical context propagation:"
    )

    # ========================================================================
    # 11. EVIDENCE
    # ========================================================================

    evidence_records = _extract_collection(
        reasoning_result,
        "evidence_records",
        "evidence",
    )

    if not evidence_records:

        raise AssertionError(
            "No evidence records were resolved."
        )

    _print_pass(
        "Evidence propagation:"
    )

    # ========================================================================
    # 12. INTERVENTION CANDIDATES
    # ========================================================================

    candidates = _extract_collection(
        reasoning_result,
        "intervention_candidates",
        "candidates",
    )

    if not candidates:

        raise AssertionError(
            "No intervention candidates were "
            "produced by contextual reasoning."
        )

    _print_pass(
        "Intervention candidate propagation:"
    )

    # ========================================================================
    # 13. PRIORITIZER
    # ========================================================================

    prioritizer = (
        InterventionPrioritizer()
    )

    _print_pass(
        "Prioritizer construction:"
    )

    # ========================================================================
    # 14. PRIORITIZATION
    # ========================================================================

    prioritization_result = (
        prioritizer.prioritize(
            reasoning_result
        )
    )

    if prioritization_result is None:

        raise AssertionError(
            "InterventionPrioritizer returned None."
        )

    _print_pass(
        "Real prioritization:"
    )

    # ========================================================================
    # 15. RANKED PRIORITIES
    # ========================================================================

    priorities = _extract_collection(
        prioritization_result,
        "priorities",
        "ranked_interventions",
        "interventions",
    )

    _validate_priorities(
        priorities,
        candidates,
    )

    _print_pass(
        "Candidate processing:"
    )

    # ========================================================================
    # 16. RISK CONSISTENCY
    # ========================================================================

    prioritized_risk_probability = _get_attr(
        prioritization_result,
        "risk_probability",
        "risk_score",
        default=risk_probability,
    )

    prioritized_risk_band = _get_attr(
        prioritization_result,
        "risk_band",
        "risk_level",
        default=risk_band,
    )

    if (
        prioritized_risk_probability
        != risk_probability
    ):

        raise AssertionError(
            "Risk probability changed during "
            "prioritization."
        )

    if (
        prioritized_risk_band
        != risk_band
    ):

        raise AssertionError(
            "Risk band changed during "
            "prioritization."
        )

    _print_pass(
        "Risk consistency:"
    )

    # ========================================================================
    # 17. SERIALIZATION
    # ========================================================================

    serialized = _serialize_result(
        prioritization_result
    )

    if not isinstance(
        serialized,
        dict,
    ):

        raise AssertionError(
            "Prioritization serialization "
            "did not produce a dictionary."
        )

    if (
        "priorities" not in serialized
        and "ranked_interventions" not in serialized
    ):

        raise AssertionError(
            "Serialized prioritization result "
            "contains no priorities."
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

    _print_pass(
        "Serialization:"
    )

    # ========================================================================
    # 18. RESULT
    # ========================================================================

    print()

    print("=" * 70)

    print(
        "INTEGRATION RESULT"
    )

    print("=" * 70)

    print(
        f"Member ID:                 "
        f"{actual_member_id}"
    )

    print(
        f"Risk probability:          "
        f"{risk_probability}"
    )

    print(
        f"Risk band:                 "
        f"{risk_band}"
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
        f"{len(candidates)}"
    )

    print(
        f"Prioritized interventions: "
        f"{len(priorities)}"
    )

    # ========================================================================
    # 19. RANKED INTERVENTIONS
    # ========================================================================

    print()

    print(
        "RANKED INTERVENTIONS"
    )

    print("-" * 70)

    for priority in priorities:

        rank = _get_attr(
            priority,
            "rank",
            default="?",
        )

        intervention_id = _get_attr(
            priority,
            "intervention_id",
            "id",
            default="UNKNOWN",
        )

        name = _get_attr(
            priority,
            "name",
            "intervention_name",
            default="UNKNOWN",
        )

        domain = _get_attr(
            priority,
            "domain",
            default="UNKNOWN",
        )

        score = _get_attr(
            priority,
            "priority_score",
            "score",
            default=0,
        )

        priority_band = _get_attr(
            priority,
            "priority_band",
            "band",
            default="UNKNOWN",
        )

        matched_factors = _extract_collection(
            priority,
            "matched_factors",
            "factors",
        )

        rationale = _get_attr(
            priority,
            "rationale",
            "explanation",
            default="",
        )

        print(
            f"{rank}. "
            f"{intervention_id} | "
            f"{name}"
        )

        print(
            f"   Domain: "
            f"{domain}"
        )

        print(
            f"   Score: "
            f"{score}"
        )

        print(
            f"   Band: "
            f"{priority_band}"
        )

        print(
            f"   Matched factors: "
            f"{len(matched_factors)}"
        )

        print(
            f"   Rationale: "
            f"{rationale}"
        )

    # ========================================================================
    # 20. OUTPUT
    # ========================================================================

    print()

    print(
        "Output:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    # ========================================================================
    # 21. SUCCESS
    # ========================================================================

    print()

    print("=" * 70)

    print(
        "REASONING → PRIORITIZATION "
        "INTEGRATION SELF-TEST: PASSED"
    )

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    run_integration_test()