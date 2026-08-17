# ============================================================================
# HEALTHLENS — LAMBDAMART → RECOMMENDATION INTEGRATION
# ============================================================================
#
# Pipeline:
#
#   Knowledge Graph
#        ↓
#   Knowledge Base Registry
#        ↓
#   Contextual Reasoner
#        ↓
#   Intervention Prioritization
#        ↓
#   LambdaMART Ranking
#        ↓
#   Recommendation Engine
#        ↓
#   Final actionable recommendations
#
# This integration intentionally supports the current HealthLens graph
# representation, where the loaded graph may be a dictionary containing
# "nodes" and "relationships".
# ============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "knowledge_graph"
    / "healthlens_kb_graph_integration.json"
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

TEST_MEMBER_ID = (
    "member:1e7909f8_39b2_3c7e_1fda_a6c3256dc061"
)


# ============================================================================
# SAFE IMPORTS
# ============================================================================

from src.reasoning.contextual_reasoner import (
    ContextualReasoner,
    load_graph,
)

from src.knowledge_base.registry import (
    build_registry,
)

from src.prioritization.intervention_prioritizer import (
    InterventionPrioritizer,
)

from src.prioritization.ranking_features import (
    RankingFeatureBuilder,
)

from src.prioritization.lambdamart_ranker import (
    LambdaMARTRanker,
)

from src.recommendations.recommendation_engine import (
    RecommendationEngine,
)


# ============================================================================
# GRAPH NORMALIZATION
# ============================================================================

def normalize_graph(graph: Any) -> dict[str, Any]:
    """
    Normalize the HealthLens graph into a dictionary representation.

    Supported inputs:

    1. dict:
       {
           "nodes": [...],
           "relationships": [...]
       }

    2. NetworkX-like object:
       graph.nodes
       graph.edges

    3. Object exposing:
       graph["nodes"]
       graph["relationships"]

    The rest of this integration uses only the normalized representation.
    """

    if graph is None:
        raise ValueError("Knowledge graph is None.")

    # ------------------------------------------------------------------
    # Already normalized dictionary
    # ------------------------------------------------------------------

    if isinstance(graph, Mapping):

        nodes = graph.get("nodes", [])

        relationships = graph.get(
            "relationships",
            graph.get("edges", []),
        )

        if nodes is None:
            nodes = []

        if relationships is None:
            relationships = []

        return {
            "nodes": list(nodes),
            "relationships": list(relationships),
        }

    # ------------------------------------------------------------------
    # NetworkX-style graph
    # ------------------------------------------------------------------

    if hasattr(graph, "nodes"):

        raw_nodes = graph.nodes

        if callable(raw_nodes):
            try:
                nodes = list(raw_nodes(data=True))
            except TypeError:
                nodes = list(raw_nodes())
        else:
            nodes = list(raw_nodes)

        relationships = []

        if hasattr(graph, "edges"):

            raw_edges = graph.edges

            if callable(raw_edges):
                try:
                    relationships = list(
                        raw_edges(data=True)
                    )
                except TypeError:
                    relationships = list(
                        raw_edges()
                    )
            else:
                relationships = list(raw_edges)

        return {
            "nodes": nodes,
            "relationships": relationships,
        }

    raise TypeError(
        "Unsupported graph type: "
        f"{type(graph).__name__}"
    )


# ============================================================================
# GRAPH ACCESSORS
# ============================================================================

def graph_nodes(graph: Any) -> list[Any]:
    """
    Return graph nodes regardless of underlying representation.
    """

    normalized = normalize_graph(graph)

    return normalized["nodes"]


def graph_relationships(graph: Any) -> list[Any]:
    """
    Return graph relationships regardless of underlying representation.
    """

    normalized = normalize_graph(graph)

    return normalized["relationships"]


# ============================================================================
# GRAPH VALIDATION
# ============================================================================

def validate_graph(graph: Any) -> dict[str, int]:
    """
    Validate graph structure.
    """

    normalized = normalize_graph(graph)

    nodes = normalized["nodes"]
    relationships = normalized["relationships"]

    if not nodes:
        raise AssertionError(
            "Knowledge Graph contains no nodes."
        )

    if not relationships:
        raise AssertionError(
            "Knowledge Graph contains no relationships."
        )

    return {
        "nodes": len(nodes),
        "relationships": len(relationships),
    }


# ============================================================================
# GRAPH LOADING
# ============================================================================

def load_healthlens_graph() -> dict[str, Any]:
    """
    Load and normalize the HealthLens knowledge graph.
    """

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            "Knowledge Graph not found:\n"
            f"{GRAPH_PATH}"
        )

    raw_graph = load_graph(GRAPH_PATH)

    graph = normalize_graph(raw_graph)

    validate_graph(graph)

    return graph


# ============================================================================
# GENERIC SERIALIZATION
# ============================================================================

def _to_dict(value: Any) -> Any:
    """
    Convert common HealthLens objects into JSON-safe dictionaries.
    """

    if value is None:
        return None

    if isinstance(value, Mapping):
        return {
            str(key): _to_dict(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _to_dict(item)
            for item in value
        ]

    if hasattr(value, "to_dict"):

        try:
            return _to_dict(
                value.to_dict()
            )
        except Exception:
            pass

    if hasattr(value, "__dataclass_fields__"):

        try:
            from dataclasses import asdict

            return _to_dict(
                asdict(value)
            )
        except Exception:
            pass

    if hasattr(value, "__dict__"):

        try:
            return {
                str(key): _to_dict(item)
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
        except Exception:
            pass

    return value


# ============================================================================
# MEMBER ID NORMALIZATION
# ============================================================================

def normalize_member_id(member_id: str) -> str:
    """
    Normalize member identifiers.

    Both forms are accepted:

        1e7909f8-39b2-3c7e-1fda-a6c3256dc061

    and:

        member:1e7909f8_39b2_3c7e_1fda_a6c3256dc061
    """

    value = str(member_id).strip()

    if value.startswith("member:"):
        return value

    if "-" in value:
        return (
            "member:"
            + value.replace("-", "_")
        )

    return value


# ============================================================================
# RESULT ACCESS HELPERS
# ============================================================================

def get_result_value(
    result: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve an attribute or dictionary value.
    """

    if result is None:
        return default

    for name in names:

        if isinstance(result, Mapping):

            if name in result:
                return result[name]

        if hasattr(result, name):

            try:
                return getattr(result, name)
            except Exception:
                continue

    return default


def get_result_list(
    result: Any,
    *names: str,
) -> list[Any]:
    """
    Safely retrieve a list-valued result field.
    """

    value = get_result_value(
        result,
        *names,
        default=[],
    )

    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return list(value)

    return [value]


# ============================================================================
# REASONING EXECUTION
# ============================================================================

def run_reasoning(
    reasoner: ContextualReasoner,
    member_id: str,
) -> Any:
    """
    Run contextual reasoning for the test member.
    """

    normalized_id = normalize_member_id(
        member_id
    )

    return reasoner.reason(
        normalized_id
    )


# ============================================================================
# LAMBDAMART FEATURE CONSTRUCTION
# ============================================================================

def build_lambda_features(
    reasoning_result: Any,
    prioritization_result: Any,
    feature_builder: LambdaMartFeatureBuilder,
) -> Any:
    """
    Construct LambdaMART ranking features.

    The existing feature builder is used rather than duplicating the
    feature engineering logic in this integration layer.
    """

    candidates = get_result_list(
        prioritization_result,
        "candidates",
        "intervention_candidates",
        "prioritized_interventions",
        "recommendations",
    )

    # Try the common feature-builder APIs used by HealthLens.
    method_names = (
        "build_features",
        "build_feature_matrix",
        "build",
        "construct_features",
    )

    last_error = None

    for method_name in method_names:

        method = getattr(
            feature_builder,
            method_name,
            None,
        )

        if method is None:
            continue

        attempts = (
            (
                reasoning_result,
                candidates,
            ),
            (
                reasoning_result,
                prioritization_result,
            ),
            (
                candidates,
            ),
        )

        for args in attempts:

            try:
                return method(*args)

            except TypeError as exc:
                last_error = exc
                continue

    raise RuntimeError(
        "Unable to construct LambdaMART features. "
        "Expected the feature builder to expose one of: "
        + ", ".join(method_names)
        + (
            f". Last error: {last_error}"
            if last_error
            else ""
        )
    )


# ============================================================================
# LAMBDAMART RANKING
# ============================================================================

def run_lambda_ranker(
    ranker: LambdaMartRanker,
    features: Any,
) -> Any:
    """
    Run the LambdaMART ranking backend.
    """

    method_names = (
        "rank",
        "predict",
        "rank_candidates",
        "predict_scores",
    )

    last_error = None

    for method_name in method_names:

        method = getattr(
            ranker,
            method_name,
            None,
        )

        if method is None:
            continue

        try:
            return method(features)

        except TypeError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Unable to execute LambdaMART ranker. "
        "Expected one of: "
        + ", ".join(method_names)
        + (
            f". Last error: {last_error}"
            if last_error
            else ""
        )
    )


# ============================================================================
# RECOMMENDATION GENERATION
# ============================================================================

def generate_recommendations(
    recommendation_engine: RecommendationEngine,
    reasoning_result: Any,
    ranked_result: Any,
) -> Any:
    """
    Generate final recommendations.

    The recommendation engine remains the authoritative component for
    recommendation wording, evidence linkage, urgency and explainability.
    """

    method_names = (
        "generate_recommendations",
        "generate",
        "recommend",
        "build_recommendations",
    )

    last_error = None

    for method_name in method_names:

        method = getattr(
            recommendation_engine,
            method_name,
            None,
        )

        if method is None:
            continue

        attempts = (
            (
                reasoning_result,
                ranked_result,
            ),
            (
                ranked_result,
            ),
            (
                reasoning_result,
            ),
        )

        for args in attempts:

            try:
                return method(*args)

            except TypeError as exc:
                last_error = exc
                continue

    raise RuntimeError(
        "Unable to generate recommendations. "
        "Expected one of: "
        + ", ".join(method_names)
        + (
            f". Last error: {last_error}"
            if last_error
            else ""
        )
    )


# ============================================================================
# OUTPUT EXTRACTION
# ============================================================================

def extract_recommendation_list(
    recommendations: Any,
) -> list[Any]:
    """
    Normalize recommendation output.
    """

    if recommendations is None:
        return []

    if isinstance(
        recommendations,
        (list, tuple, set),
    ):
        return list(recommendations)

    if isinstance(
        recommendations,
        Mapping,
    ):

        for key in (
            "recommendations",
            "results",
            "items",
            "ranked_recommendations",
        ):

            if key in recommendations:

                value = recommendations[key]

                if isinstance(
                    value,
                    (list, tuple, set),
                ):
                    return list(value)

        return [
            recommendations
        ]

    for name in (
        "recommendations",
        "results",
        "items",
        "ranked_recommendations",
    ):

        value = getattr(
            recommendations,
            name,
            None,
        )

        if value is not None:

            if isinstance(
                value,
                (list, tuple, set),
            ):
                return list(value)

    return [recommendations]


# ============================================================================
# VALIDATION
# ============================================================================

def validate_recommendations(
    recommendations: Any,
) -> None:
    """
    Validate final recommendation output.
    """

    items = extract_recommendation_list(
        recommendations
    )

    if not items:
        raise AssertionError(
            "Recommendation engine returned no recommendations."
        )

    for index, item in enumerate(items, start=1):

        intervention_id = get_result_value(
            item,
            "intervention_id",
            "id",
            default=None,
        )

        if not intervention_id:

            # Some recommendation implementations wrap
            # intervention information.
            intervention = get_result_value(
                item,
                "intervention",
                default=None,
            )

            intervention_id = get_result_value(
                intervention,
                "intervention_id",
                "id",
                default=None,
            )

        if not intervention_id:
            raise AssertionError(
                "Recommendation "
                f"{index} has no intervention ID."
            )


# ============================================================================
# SERIALIZATION
# ============================================================================

def build_output(
    member_id: str,
    graph_stats: Mapping[str, int],
    reasoning_result: Any,
    prioritization_result: Any,
    ranked_result: Any,
    recommendations: Any,
) -> dict[str, Any]:
    """
    Build the complete integration artifact.
    """

    return {
        "pipeline": (
            "Knowledge Graph → "
            "Contextual Reasoning → "
            "Prioritization → "
            "LambdaMART → "
            "Recommendation"
        ),
        "member_id": member_id,
        "graph": {
            "nodes": graph_stats["nodes"],
            "relationships": graph_stats[
                "relationships"
            ],
        },
        "reasoning": _to_dict(
            reasoning_result
        ),
        "prioritization": _to_dict(
            prioritization_result
        ),
        "lambdamart": _to_dict(
            ranked_result
        ),
        "recommendations": _to_dict(
            recommendations
        ),
    }


def write_output(
    output: Mapping[str, Any],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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


# ============================================================================
# ENGINE CONSTRUCTION
# ============================================================================

def construct_reasoner(
    graph: Any,
    registry: Any,
) -> ContextualReasoner:
    """
    Construct ContextualReasoner while supporting the current constructor.
    """

    attempts = (
        {
            "graph": graph,
            "registry": registry,
        },
        {
            "knowledge_graph": graph,
            "registry": registry,
        },
        {
            "graph": graph,
        },
        {
            "knowledge_graph": graph,
        },
    )

    last_error = None

    for kwargs in attempts:

        try:

            return ContextualReasoner(
                **kwargs
            )

        except TypeError as exc:

            last_error = exc
            continue

    raise RuntimeError(
        "Unable to construct ContextualReasoner. "
        f"Last error: {last_error}"
    )


def construct_prioritizer(
    registry: Any,
) -> InterventionPrioritizer:

    attempts = (
        {"registry": registry},
        {},
    )

    last_error = None

    for kwargs in attempts:

        try:

            return InterventionPrioritizer(
                **kwargs
            )

        except TypeError as exc:

            last_error = exc
            continue

    raise RuntimeError(
        "Unable to construct InterventionPrioritizer. "
        f"Last error: {last_error}"
    )


def construct_feature_builder() -> LambdaMartFeatureBuilder:

    try:
        return LambdaMartFeatureBuilder()
    except TypeError:
        return LambdaMartFeatureBuilder(
            version="1.0.0"
        )


def construct_ranker() -> LambdaMartRanker:

    try:
        return LambdaMartRanker()
    except TypeError:
        return LambdaMartRanker(
            backend="deterministic_baseline"
        )


def construct_recommendation_engine(
    registry: Any,
) -> RecommendationEngine:

    attempts = (
        {"registry": registry},
        {},
    )

    last_error = None

    for kwargs in attempts:

        try:

            return RecommendationEngine(
                **kwargs
            )

        except TypeError as exc:

            last_error = exc
            continue

    raise RuntimeError(
        "Unable to construct RecommendationEngine. "
        f"Last error: {last_error}"
    )


# ============================================================================
# PRIORITIZATION
# ============================================================================

def run_prioritization(
    prioritizer: InterventionPrioritizer,
    reasoning_result: Any,
) -> Any:
    """
    Execute existing intervention prioritization.
    """

    method_names = (
        "prioritize",
        "rank",
        "run",
    )

    last_error = None

    for method_name in method_names:

        method = getattr(
            prioritizer,
            method_name,
            None,
        )

        if method is None:
            continue

        attempts = (
            (reasoning_result,),
            (
                get_result_list(
                    reasoning_result,
                    "intervention_candidates",
                    "candidates",
                ),
                reasoning_result,
            ),
            (
                get_result_list(
                    reasoning_result,
                    "intervention_candidates",
                    "candidates",
                ),
            ),
        )

        for args in attempts:

            try:
                return method(*args)

            except TypeError as exc:
                last_error = exc
                continue

    raise RuntimeError(
        "Unable to execute intervention prioritization. "
        f"Last error: {last_error}"
    )


# ============================================================================
# MAIN INTEGRATION TEST
# ============================================================================

def run_integration_test() -> dict[str, Any]:

    print("=" * 78)
    print(
        "HEALTHLENS LAMBDAMART → "
        "RECOMMENDATION INTEGRATION"
    )
    print("=" * 78)

    # ------------------------------------------------------------------
    # GRAPH
    # ------------------------------------------------------------------

    graph = load_healthlens_graph()

    graph_stats = validate_graph(
        graph
    )

    print(
        "Test member:                         "
        f"{TEST_MEMBER_ID}"
    )

    print(
        "Knowledge Graph:                      PASS"
    )

    print(
        "Graph nodes:                          "
        f"{graph_stats['nodes']}"
    )

    print(
        "Graph relationships:                  "
        f"{graph_stats['relationships']}"
    )

    # ------------------------------------------------------------------
    # REGISTRY
    # ------------------------------------------------------------------

    registry = build_registry()

    print(
        "Knowledge Base Registry:              PASS"
    )

    # ------------------------------------------------------------------
    # REASONER
    # ------------------------------------------------------------------

    reasoner = construct_reasoner(
        graph,
        registry,
    )

    print(
        "Contextual Reasoner:                  PASS"
    )

    # ------------------------------------------------------------------
    # MEMBER REASONING
    # ------------------------------------------------------------------

    reasoning_result = run_reasoning(
        reasoner,
        TEST_MEMBER_ID,
    )

    print(
        "Member reasoning:                     PASS"
    )

    risk_probability = get_result_value(
        reasoning_result,
        "risk_probability",
        default=None,
    )

    risk_band = get_result_value(
        reasoning_result,
        "risk_band",
        default=None,
    )

    sdoh_factors = get_result_list(
        reasoning_result,
        "sdoh_factors",
        "relevant_sdoh_factors",
    )

    sdoh_domains = get_result_list(
        reasoning_result,
        "sdoh_domains",
        "domains",
    )

    clinical_factors = get_result_list(
        reasoning_result,
        "clinical_factors",
    )

    evidence = get_result_list(
        reasoning_result,
        "evidence_records",
        "evidence",
    )

    candidates = get_result_list(
        reasoning_result,
        "intervention_candidates",
        "candidates",
    )

    print(
        "Risk probability:                    "
        f"{risk_probability}"
    )

    print(
        "Risk band:                           "
        f"{risk_band}"
    )

    print(
        "SDOH factors:                        "
        f"{len(sdoh_factors)}"
    )

    print(
        "SDOH domains:                        "
        f"{len(sdoh_domains)}"
    )

    print(
        "Clinical factors:                    "
        f"{len(clinical_factors)}"
    )

    print(
        "Evidence records:                    "
        f"{len(evidence)}"
    )

    print(
        "Intervention candidates:             "
        f"{len(candidates)}"
    )

    # ------------------------------------------------------------------
    # PRIORITIZATION
    # ------------------------------------------------------------------

    prioritizer = construct_prioritizer(
        registry
    )

    print(
        "Intervention Prioritizer:            PASS"
    )

    prioritization_result = run_prioritization(
        prioritizer,
        reasoning_result,
    )

    print(
        "Intervention prioritization:         PASS"
    )

    # ------------------------------------------------------------------
    # LAMBDAMART FEATURES
    # ------------------------------------------------------------------

    feature_builder = (
        construct_feature_builder()
    )

    print(
        "LambdaMART Feature Builder:           PASS"
    )

    features = build_lambda_features(
        reasoning_result,
        prioritization_result,
        feature_builder,
    )

    if features is None:
        raise AssertionError(
            "LambdaMART feature construction returned None."
        )

    print(
        "LambdaMART feature construction:      PASS"
    )

    # ------------------------------------------------------------------
    # LAMBDAMART
    # ------------------------------------------------------------------

    ranker = construct_ranker()

    print(
        "LambdaMART Ranker:                    PASS"
    )

    ranked_result = run_lambda_ranker(
        ranker,
        features,
    )

    if ranked_result is None:
        raise AssertionError(
            "LambdaMART ranker returned None."
        )

    print(
        "LambdaMART ranking:                   PASS"
    )

    # ------------------------------------------------------------------
    # RECOMMENDATION ENGINE
    # ------------------------------------------------------------------

    recommendation_engine = (
        construct_recommendation_engine(
            registry
        )
    )

    print(
        "Recommendation Engine:               PASS"
    )

    recommendations = generate_recommendations(
        recommendation_engine,
        reasoning_result,
        ranked_result,
    )

    validate_recommendations(
        recommendations
    )

    recommendation_list = (
        extract_recommendation_list(
            recommendations
        )
    )

    print(
        "Recommendation generation:           PASS"
    )

    print(
        "Recommendation validation:            PASS"
    )

    # ------------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------------

    output = build_output(
        member_id=normalize_member_id(
            TEST_MEMBER_ID
        ),
        graph_stats=graph_stats,
        reasoning_result=reasoning_result,
        prioritization_result=prioritization_result,
        ranked_result=ranked_result,
        recommendations=recommendations,
    )

    write_output(
        output
    )

    print(
        "Recommendation serialization:         PASS"
    )

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "LAMBDAMART → RECOMMENDATION RESULT"
    )
    print("=" * 78)

    print(
        "Member ID:                 "
        f"{normalize_member_id(TEST_MEMBER_ID)}"
    )

    print(
        "Risk probability:          "
        f"{risk_probability}"
    )

    print(
        "Risk band:                 "
        f"{risk_band}"
    )

    print(
        "Recommendations:           "
        f"{len(recommendation_list)}"
    )

    print()

    print(
        "ACTIONABLE RECOMMENDATIONS"
    )

    print("-" * 78)

    for index, recommendation in enumerate(
        recommendation_list,
        start=1,
    ):

        intervention_id = get_result_value(
            recommendation,
            "intervention_id",
            "id",
            default="UNKNOWN",
        )

        intervention_name = get_result_value(
            recommendation,
            "intervention_name",
            "name",
            "title",
            default="",
        )

        priority_score = get_result_value(
            recommendation,
            "priority_score",
            "score",
            "ranking_score",
            default=None,
        )

        priority_band = get_result_value(
            recommendation,
            "priority_band",
            "band",
            default=None,
        )

        urgency = get_result_value(
            recommendation,
            "urgency",
            default=None,
        )

        print(
            f"{index}. "
            f"{intervention_id}"
            + (
                f" | {intervention_name}"
                if intervention_name
                else ""
            )
        )

        if priority_score is not None:
            print(
                f"   Priority score: {priority_score}"
            )

        if priority_band is not None:
            print(
                f"   Priority band:  {priority_band}"
            )

        if urgency is not None:
            print(
                f"   Urgency:        {urgency}"
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
        "LAMBDAMART → RECOMMENDATION "
        "INTEGRATION SELF-TEST: PASSED"
    )
    print("=" * 78)

    return output


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_integration_test()