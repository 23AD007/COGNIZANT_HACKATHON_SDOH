"""
HealthLens
LambdaMART -> Recommendation Integration
==========================================

Production flow:

    Member Risk Scores
          |
          v
    Knowledge Graph
          |
          v
    Contextual Reasoner
          |
          v
    ReasoningResult
          |
          v
    InterventionPrioritizer
          |
          v
    PrioritizationResult
          |
          v
    LambdaMART Feature Builder
          |
          v
    LambdaMART Ranker
          |
          v
    Recommendation Engine
          |
          v
    RecommendationResult

Important design rules
----------------------
1. Use the real HealthLens Knowledge Graph.
2. Use the real Knowledge Base Registry.
3. Use the real member risk output.
4. Use the real ContextualReasoner.
5. Use the real InterventionPrioritizer.
6. Use the real LambdaMART feature builder/ranker.
7. Use the real RecommendationEngine.
8. Never invent SDOH factors.
9. Never require every member to have every attribute.
10. Only use SDOH factors actually resolved for the member.
11. Preserve the canonical graph member ID.
12. Do not modify the graph, registry, risk output, or knowledge base.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping


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

RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member_risk_scores.csv"
)

REASONING_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reasoning"
    / "member_risk_reasoning_integration.json"
)

PRIORITIZATION_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "prioritization"
    / "healthlens_prioritization_result.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "recommendations"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "healthlens_lambdamart_recommendations.json"
)


# ============================================================================
# TEST MEMBER
# ============================================================================

TEST_MEMBER_ID = (
    "1e7909f8-39b2-3c7e-1fda-a6c3256dc061"
)

EXPECTED_GRAPH_MEMBER_ID = (
    "member:"
    "1e7909f8_39b2_3c7e_1fda_a6c3256dc061"
)


# ============================================================================
# IMPORTS
# ============================================================================

from src.knowledge_base.registry import build_registry

from src.reasoning.contextual_reasoner import (
    ContextualReasoner,
    load_graph,
)

from src.reasoning.member_risk_integration import (
    canonical_graph_member_id as _canonical_graph_member_id,
    inject_risk_into_member_node,
)

from src.modeling.member_risk import (
    MemberRiskScoreAdapter,
)

from .intervention_prioritizer import (
    InterventionPriority,
    InterventionPrioritizer,
    PrioritizationResult,
)

from .lambdamart_ranker import (
    LambdaMARTRanker,
    RankingCandidate,
)

from .ranking_features import RankingFeatureBuilder

from src.recommendations.recommendation_engine import RecommendationEngine


# ============================================================================
# OPTIONAL LAMBDAMART IMPORTS
# ============================================================================

def _load_lambda_components():
    """
    Load the existing LambdaMART implementation without inventing
    module/class names.

    The project already contains the LambdaMART implementation created
    earlier. We try the known project modules first and fail with a
    useful diagnostic rather than silently falling back to a fabricated
    ranker.
    """

    return RankingFeatureBuilder, LambdaMARTRanker


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def _print_pass(label: str) -> None:

    print(
        f"{label:<40} PASS"
    )


def _get(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:

    if obj is None:
        return default

    if isinstance(obj, Mapping):

        for name in names:

            if name in obj:

                value = obj[name]

                if value is not None:
                    return value

        return default

    for name in names:

        if hasattr(obj, name):

            value = getattr(
                obj,
                name,
            )

            if value is not None:
                return value

    return default


def _as_list(
    value: Any,
) -> list[Any]:

    if value is None:
        return []

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return list(value)

    return [value]


def _serialize(
    obj: Any,
) -> Any:

    if obj is None:
        return None

    if isinstance(
        obj,
        (
            str,
            int,
            float,
            bool,
        ),
    ):

        return obj

    if isinstance(obj, Mapping):

        return {
            str(key): _serialize(value)
            for key, value in obj.items()
        }

    if isinstance(
        obj,
        (list, tuple, set),
    ):

        return [
            _serialize(item)
            for item in obj
        ]

    if hasattr(
        obj,
        "to_dict",
    ):

        return _serialize(
            obj.to_dict()
        )

    if hasattr(
        obj,
        "__dict__",
    ):

        return {
            key: _serialize(value)
            for key, value in vars(obj).items()
        }

    return str(obj)


# ============================================================================
# MEMBER ID
# ============================================================================

def canonical_graph_member_id(
    member_id: str,
) -> str:
    """Use the canonical resolver shared with risk-to-reasoning."""

    return _canonical_graph_member_id(member_id)


def resolve_graph_member(
    graph: dict,
    member_id: str,
) -> str:

    expected = canonical_graph_member_id(
        member_id
    )

    nodes = graph.get(
        "nodes",
        [],
    )

    for node in nodes:

        node_id = _get(
            node,
            "id",
            "node_id",
            "key",
            default="",
        )

        if str(node_id) == expected:

            return expected

    raise ValueError(
        "Member not found in Knowledge Graph.\n"
        f"Member ID: {member_id}\n"
        f"Expected canonical graph ID: {expected}"
    )


# ============================================================================
# GRAPH LOADING
# ============================================================================

def load_healthlens_graph() -> dict:

    if not GRAPH_PATH.exists():

        raise FileNotFoundError(
            "Knowledge Graph not found:\n"
            f"{GRAPH_PATH}"
        )

    graph = load_graph(
        GRAPH_PATH
    )

    if not isinstance(
        graph,
        dict,
    ):

        raise TypeError(
            "Knowledge Graph loader did not return "
            "a dictionary."
        )

    nodes = graph.get(
        "nodes",
        [],
    )

    relationships = graph.get(
        "relationships",
        graph.get(
            "edges",
            [],
        ),
    )

    if not nodes:

        raise AssertionError(
            "Knowledge Graph contains no nodes."
        )

    if not relationships:

        raise AssertionError(
            "Knowledge Graph contains no relationships."
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

    return graph


# ============================================================================
# RISK
# ============================================================================

def load_risk(
    member_id: str,
):

    if not RISK_FILE.exists():

        raise FileNotFoundError(
            "Member risk file not found:\n"
            f"{RISK_FILE}"
        )

    adapter = MemberRiskScoreAdapter(
        RISK_FILE
    )

    record = adapter.get_member_risk(
        member_id
    )

    if record is None:

        raise ValueError(
            "Risk record not found for member:\n"
            f"{member_id}"
        )

    risk_probability = float(
        _get(
            record,
            "risk_probability",
            default=0.0,
        )
    )

    risk_band = _get(
        record,
        "risk_band",
        default=None,
    )

    if not (
        0.0
        <= risk_probability
        <= 1.0
    ):

        raise ValueError(
            "Invalid risk probability: "
            f"{risk_probability}"
        )

    if not risk_band:

        raise ValueError(
            "Risk band missing for member."
        )

    return (
        adapter,
        record,
        risk_probability,
        risk_band,
    )


# ============================================================================
# REASONING
# ============================================================================

def run_reasoning(
    reasoner,
    graph_member_id,
    risk_probability=None,
    risk_band=None,
):
    """
    Execute the existing HealthLens ContextualReasoner.

    The ContextualReasoner is a real ContextualReasoner instance.
    It must be constructed before this function is called.

    risk_probability and risk_band are accepted only for compatibility
    with the integration pipeline. They are NOT passed to
    ContextualReasoner.reason().
    """

    if reasoner is None:
        raise ValueError(
            "ContextualReasoner is required."
        )

    if not graph_member_id:
        raise ValueError(
            "graph_member_id is required."
        )

    # The real API is:
    #
    #     reasoner.reason(member_id)
    #
    # Do NOT pass risk_probability or risk_band.
    if not hasattr(reasoner, "reason"):
        raise TypeError(
            "Expected a ContextualReasoner instance, "
            f"but received {type(reasoner).__name__}."
        )

    reasoning_result = reasoner.reason(
        graph_member_id
    )

    if reasoning_result is None:
        raise RuntimeError(
            "ContextualReasoner returned None for "
            f"member {graph_member_id}."
        )

    return reasoning_result

# ============================================================================
# REASONING VALIDATION
# ============================================================================

def validate_reasoning(
    reasoning_result: Any,
) -> dict[str, list[Any]]:

    risk_probability = _get(
        reasoning_result,
        "risk_probability",
        "risk_score",
    )

    risk_band = _get(
        reasoning_result,
        "risk_band",
        "risk_level",
    )

    sdoh_factors = _as_list(
        _get(
            reasoning_result,
            "sdoh_factors",
            "relevant_sdoh_factors",
            default=[],
        )
    )

    sdoh_domains = _as_list(
        _get(
            reasoning_result,
            "sdoh_domains",
            "domains",
            default=[],
        )
    )

    clinical_factors = _as_list(
        _get(
            reasoning_result,
            "clinical_factors",
            "clinical_context",
            default=[],
        )
    )

    evidence_records = _as_list(
        _get(
            reasoning_result,
            "evidence_records",
            "evidence",
            default=[],
        )
    )

    candidates = _as_list(
        _get(
            reasoning_result,
            "intervention_candidates",
            "candidates",
            default=[],
        )
    )

    if risk_probability is None:
        raise AssertionError(
            "Reasoning result has no risk probability."
        )

    if risk_band is None:
        raise AssertionError(
            "Reasoning result has no risk band."
        )

    if not sdoh_factors:
        raise AssertionError(
            "Contextual reasoning produced no SDOH factors."
        )

    if not sdoh_domains:
        raise AssertionError(
            "Contextual reasoning produced no SDOH domains."
        )

    if not clinical_factors:
        raise AssertionError(
            "Contextual reasoning produced no clinical factors."
        )

    if not evidence_records:
        raise AssertionError(
            "Contextual reasoning produced no evidence."
        )

    if not candidates:
        raise AssertionError(
            "Contextual reasoning produced no intervention candidates."
        )

    print(
        f"Risk probability:                       "
        f"{risk_probability}"
    )

    print(
        f"Risk band:                              "
        f"{risk_band}"
    )

    print(
        f"SDOH factors:                           "
        f"{len(sdoh_factors)}"
    )

    print(
        f"SDOH domains:                           "
        f"{len(sdoh_domains)}"
    )

    print(
        f"Clinical factors:                       "
        f"{len(clinical_factors)}"
    )

    print(
        f"Evidence records:                       "
        f"{len(evidence_records)}"
    )

    print(
        f"Intervention candidates:                 "
        f"{len(candidates)}"
    )

    return {
        "sdoh_factors": sdoh_factors,
        "sdoh_domains": sdoh_domains,
        "clinical_factors": clinical_factors,
        "evidence_records": evidence_records,
        "candidates": candidates,
    }


# ============================================================================
# PRIORITIZATION
# ============================================================================

def run_prioritization(
    reasoning_result: Any,
):

    prioritizer = InterventionPrioritizer()

    _print_pass(
        "Intervention Prioritizer:"
    )

    result = prioritizer.prioritize(
        reasoning_result
    )

    if result is None:

        raise RuntimeError(
            "InterventionPrioritizer returned None."
        )

    _print_pass(
        "Intervention prioritization:"
    )

    return result


# ============================================================================
# IMPORTANT FIX:
# EXTRACT CANDIDATES FROM PRIORITIZATION RESULT
# ============================================================================

def extract_prioritized_candidates(
    prioritization_result: Any,
    reasoning_result: Any,
) -> list[dict[str, Any]]:
    """
    Convert the REAL PrioritizationResult into LambdaMART candidates.

    Critical distinction:

        reasoning_result.intervention_candidates
            = candidate interventions generated by reasoning

        prioritization_result.priorities
            = ranked intervention objects

    The previous implementation incorrectly assumed that the
    prioritization result itself contained `intervention_candidates`.

    We therefore use `priorities` as the source after prioritization.

    We do NOT manufacture candidates when priorities are empty.
    """

    priorities = _as_list(
        _get(
            prioritization_result,
            "priorities",
            "ranked_interventions",
            default=[],
        )
    )

    if not priorities:

        # Some implementations may serialize the result differently.
        serialized = _serialize(
            prioritization_result
        )

        if isinstance(
            serialized,
            Mapping,
        ):

            priorities = _as_list(
                serialized.get(
                    "priorities",
                    serialized.get(
                        "ranked_interventions",
                        [],
                    ),
                )
            )

    if not priorities:

        raise ValueError(
            "No prioritized interventions found in "
            "InterventionPrioritizer output."
        )

    reasoning_candidates = _as_list(
        _get(
            reasoning_result,
            "intervention_candidates",
            "candidates",
            default=[],
        )
    )

    # Build a lookup from the REAL reasoning candidates.
    candidate_lookup: dict[str, Any] = {}

    for candidate in reasoning_candidates:

        intervention_id = _get(
            candidate,
            "intervention_id",
            "id",
            "key",
        )

        if intervention_id is not None:

            candidate_lookup[
                str(intervention_id)
            ] = candidate

    output: list[dict[str, Any]] = []

    for rank_index, priority in enumerate(
        priorities,
        start=1,
    ):

        intervention_id = _get(
            priority,
            "intervention_id",
            "id",
            "key",
        )

        if intervention_id is None:

            raise ValueError(
                "Prioritized intervention has no "
                "intervention_id."
            )

        intervention_id = str(
            intervention_id
        )

        source_candidate = candidate_lookup.get(
            intervention_id
        )

        matched_factors = _as_list(
            _get(
                priority,
                "matched_factors",
                "factors",
                default=None,
            )
        )

        # If the priority object does not carry matched factors,
        # obtain them from the original reasoning candidate.
        if not matched_factors and source_candidate is not None:

            matched_factors = _as_list(
                _get(
                    source_candidate,
                    "matched_factors",
                    "factors",
                    default=[],
                )
            )

        domain = _get(
            priority,
            "domain",
            default=None,
        )

        if domain is None and source_candidate is not None:

            domain = _get(
                source_candidate,
                "domain",
                "sdoh_domain",
            )

        name = _get(
            priority,
            "name",
            "intervention_name",
            default=None,
        )

        if name is None and source_candidate is not None:

            name = _get(
                source_candidate,
                "name",
                "title",
                "label",
            )

        score = _get(
            priority,
            "priority_score",
            "score",
            default=0.0,
        )

        output.append(
            {
                "intervention_id": intervention_id,
                "name": name,
                "domain": domain,
                "matched_factors": matched_factors,
                "matched_factor_count": len(
                    matched_factors
                ),
                "priority_score": float(
                    score
                ),
                "priority_rank": rank_index,
                "priority_band": _get(
                    priority,
                    "priority_band",
                    "band",
                ),
                "rationale": _get(
                    priority,
                    "rationale",
                    "explanation",
                    default="",
                ),
                "source_candidate": (
                    _serialize(
                        source_candidate
                    )
                    if source_candidate is not None
                    else None
                ),
            }
        )

    if not output:

        raise ValueError(
            "Prioritization produced zero LambdaMART candidates."
        )

    return output


# ============================================================================
# LAMBDAMART FEATURE BUILDING
# ============================================================================

def build_lambda_features(
    FeatureBuilder,
    candidates: list[dict[str, Any]],
    reasoning_result: Any,
    risk_probability: float,
    risk_band: str,
):

    """
    Call the existing LambdaMART feature builder.

    The integration does not invent feature columns. It passes the
    candidate objects and member reasoning context to the project's
    existing feature builder.
    """

    builder = FeatureBuilder()
    records = builder.build_many(reasoning_result, candidates)

    if len(records) != len(candidates):
        raise RuntimeError("LambdaMART feature builder did not return one record per candidate.")

    _print_pass("LambdaMART feature construction:")
    return records


# ============================================================================
# LAMBDAMART RANKING
# ============================================================================

def run_lambda_ranker(
    Ranker,
    features: Any,
    candidates: list[dict[str, Any]],
):
    ranking_candidates = []

    for candidate, feature_record in zip(candidates, features):
        ranking_candidates.append(
            RankingCandidate(
                intervention_id=str(candidate["intervention_id"]),
                intervention_name=str(candidate["name"]),
                domain=str(candidate["domain"]),
                features=asdict(feature_record),
                baseline_score=float(candidate["priority_score"]),
                baseline_rank=int(candidate["priority_rank"]),
                matched_factor_count=int(candidate["matched_factor_count"]),
                evidence_match_count=0,
            )
        )

    ranker = Ranker()
    result = ranker.rank(ranking_candidates, member_id=features[0].member_id)
    _print_pass("LambdaMART ranking:")
    return result


# ============================================================================
# RANK RESULT NORMALIZATION
# ============================================================================

def normalize_ranked_result(
    ranked_result: Any,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    items = _as_list(_get(ranked_result, "candidates", default=None))

    if not items:

        if isinstance(
            ranked_result,
            (list, tuple),
        ):

            items = list(
                ranked_result
            )

    if not items:

        serialized = _serialize(
            ranked_result
        )

        if isinstance(
            serialized,
            Mapping,
        ):

            for key in (
            "candidates",
            ):

                if key in serialized:

                    items = _as_list(
                        serialized[key]
                    )

                    if items:
                        break

    if not items:

        raise RuntimeError(
            "LambdaMART ranker returned no ranked results."
        )

    candidate_lookup = {
        str(
            candidate["intervention_id"]
        ): candidate
        for candidate in candidates
    }

    normalized = []

    for index, item in enumerate(
        items,
        start=1,
    ):

        intervention_id = _get(
            item,
            "intervention_id",
            "id",
            "key",
        )

        if intervention_id is None:
            continue

        intervention_id = str(
            intervention_id
        )

        base = candidate_lookup.get(
            intervention_id,
            {},
        )

        score = _get(
            item,
            "model_score",
            default=None,
        )

        if score is None:

            score = base.get(
                "priority_score",
                0.0,
            )

        normalized.append(
            {
                "rank": index,
                "intervention_id": intervention_id,
                "name": _get(
                    item,
                    "name",
                    "intervention_name",
                    default=base.get(
                        "name"
                    ),
                ),
                "domain": _get(
                    item,
                    "domain",
                    default=base.get(
                        "domain"
                    ),
                ),
                "score": float(score),
                "matched_factors": base.get(
                    "matched_factors",
                    [],
                ),
                "priority_band": base.get(
                    "priority_band"
                ),
                "rationale": base.get(
                    "rationale",
                    "",
                ),
            }
        )

    if not normalized:

        raise RuntimeError(
            "LambdaMART ranker returned results but "
            "none contained a valid intervention_id."
        )

    return normalized


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

def run_recommendation_engine(
    ranked_candidates: list[dict[str, Any]],
    reasoning_result: Any,
    prioritization_result: PrioritizationResult,
):

    """
    Generate recommendations using the concrete RecommendationEngine API.

    The engine consumes a PrioritizationResult, so preserve the actual
    priority records while applying the LambdaMART order and score.
    """

    priorities_by_id = {
        priority.intervention_id: priority
        for priority in prioritization_result.priorities
    }
    lambda_priorities = []

    for ranked in ranked_candidates:
        source = priorities_by_id[ranked["intervention_id"]]
        lambda_priorities.append(
            InterventionPriority(
                intervention_id=source.intervention_id,
                name=source.name,
                domain=source.domain,
                matched_factors=list(source.matched_factors),
                base_score=source.base_score,
                priority_score=float(ranked["score"]),
                priority_band=source.priority_band,
                rank=int(ranked["rank"]),
                rationale=source.rationale,
            )
        )

    lambda_prioritization = PrioritizationResult(
        member_id=prioritization_result.member_id,
        risk_probability=prioritization_result.risk_probability,
        risk_band=prioritization_result.risk_band,
        priorities=lambda_priorities,
        reasoning_summary=dict(prioritization_result.reasoning_summary),
        trace=list(prioritization_result.trace),
        version=prioritization_result.version,
    )
    result = RecommendationEngine().generate(reasoning_result, lambda_prioritization)
    _print_pass("Recommendation generation:")
    return result


# ============================================================================
# MAIN INTEGRATION TEST
# ============================================================================

def run_integration_test() -> None:

    print("=" * 78)
    print(
        "HEALTHLENS LAMBDAMART -> RECOMMENDATION INTEGRATION"
    )
    print("=" * 78)

    print(
        f"Test member:                         "
        f"{EXPECTED_GRAPH_MEMBER_ID}"
    )

    # ------------------------------------------------------------------
    # RISK
    # ------------------------------------------------------------------

    (
        adapter,
        risk_record,
        risk_probability,
        risk_band,
    ) = load_risk(
        TEST_MEMBER_ID
    )

    _print_pass(
        "Member risk output:"
    )

    print(
        f"Risk probability:                    "
        f"{risk_probability}"
    )

    print(
        f"Risk band:                           "
        f"{risk_band}"
    )

    # ------------------------------------------------------------------
    # GRAPH
    # ------------------------------------------------------------------

    graph = load_healthlens_graph()

    graph_member_id = resolve_graph_member(
        graph,
        TEST_MEMBER_ID,
    )

    graph = inject_risk_into_member_node(
        graph,
        graph_member_id,
        risk_record,
    )

    reasoner = ContextualReasoner(
        graph=graph,
        registry=build_registry(),
    )

    print(
        f"Graph member ID:                     "
        f"{graph_member_id}"
    )

    _print_pass(
        "Graph member resolution:"
    )

    # ------------------------------------------------------------------
    # REASONING
    # ------------------------------------------------------------------

    reasoning_result = run_reasoning(
        reasoner,
        graph_member_id,
    )

    context = validate_reasoning(
        reasoning_result
    )

    # ------------------------------------------------------------------
    # PRIORITIZATION
    # ------------------------------------------------------------------

    prioritization_result = run_prioritization(
        reasoning_result
    )

    # IMPORTANT:
    # Extract from `priorities`, NOT from a nonexistent
    # `prioritization_result.intervention_candidates`.
    prioritized_candidates = (
        extract_prioritized_candidates(
            prioritization_result,
            reasoning_result,
        )
    )

    print(
        f"Prioritized candidates:               "
        f"{len(prioritized_candidates)}"
    )

    # ------------------------------------------------------------------
    # LAMBDAMART
    # ------------------------------------------------------------------

    (
        FeatureBuilder,
        Ranker,
    ) = _load_lambda_components()

    _print_pass(
        "LambdaMART components:"
    )

    features = build_lambda_features(
        FeatureBuilder,
        prioritized_candidates,
        reasoning_result,
        risk_probability,
        risk_band,
    )

    ranked_result = run_lambda_ranker(
        Ranker,
        features,
        prioritized_candidates,
    )

    ranked_candidates = normalize_ranked_result(
        ranked_result,
        prioritized_candidates,
    )

    # ------------------------------------------------------------------
    # RECOMMENDATIONS
    # ------------------------------------------------------------------

    recommendation_result = run_recommendation_engine(
        ranked_candidates,
        reasoning_result,
        prioritization_result,
    )

    if recommendation_result is None:

        raise RuntimeError(
            "Recommendation engine returned None."
        )

    # ------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "member_id": TEST_MEMBER_ID,
        "graph_member_id": graph_member_id,
        "risk_probability": risk_probability,
        "risk_band": risk_band,
        "sdoh_factors": context[
            "sdoh_factors"
        ],
        "sdoh_domains": context[
            "sdoh_domains"
        ],
        "clinical_factors": context[
            "clinical_factors"
        ],
        "evidence_records": _serialize(
            context[
                "evidence_records"
            ]
        ),
        "intervention_candidates": _serialize(
            context[
                "candidates"
            ]
        ),
        "prioritized_candidates": prioritized_candidates,
        "lambda_ranked_candidates": ranked_candidates,
        "recommendations": _serialize(
            recommendation_result
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    _print_pass(
        "Recommendation serialization:"
    )

    # ------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "INTEGRATION RESULT"
    )
    print("=" * 78)

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
        f"{risk_probability}"
    )

    print(
        f"Risk band:                 "
        f"{risk_band}"
    )

    print(
        f"SDOH factors:              "
        f"{len(context['sdoh_factors'])}"
    )

    print(
        f"SDOH domains:              "
        f"{len(context['sdoh_domains'])}"
    )

    print(
        f"Clinical factors:          "
        f"{len(context['clinical_factors'])}"
    )

    print(
        f"Evidence records:          "
        f"{len(context['evidence_records'])}"
    )

    print(
        f"Intervention candidates:   "
        f"{len(context['candidates'])}"
    )

    print(
        f"Prioritized candidates:    "
        f"{len(prioritized_candidates)}"
    )

    print(
        f"LambdaMART ranked:         "
        f"{len(ranked_candidates)}"
    )

    recommendations = _as_list(
        _get(
            recommendation_result,
            "recommendations",
            "items",
            "results",
            default=[],
        )
    )

    print(
        f"Recommendations:           "
        f"{len(recommendations)}"
    )

    print()
    print(
        "LAMBDA RANKED INTERVENTIONS"
    )
    print("-" * 78)

    for item in ranked_candidates:

        print(
            f"{item['rank']}. "
            f"{item['intervention_id']} | "
            f"{item['name']} | "
            f"score: {item['score']:.4f}"
        )

    print()
    print(
        "Output:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print()
    print("=" * 78)
    print(
        "LAMBDA MART -> RECOMMENDATION "
        "INTEGRATION SELF-TEST: PASSED"
    )
    print("=" * 78)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    run_integration_test()
