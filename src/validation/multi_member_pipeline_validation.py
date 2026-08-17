"""
HealthLens
Multi-Member End-to-End Pipeline Validation

Validates the existing production pipeline across real members:

    member_risk_scores.csv
            |
            v
    MemberRiskScoreAdapter
            |
            v
    canonical graph member
            |
            v
    Knowledge Graph
            |
            v
    ContextualReasoner
            |
            v
    InterventionPrioritizer
            |
            v
    RankingFeatureBuilder
            |
            v
    LambdaMARTRanker
            |
            v
    RecommendationEngine

Design requirements:
- use real members
- use existing project APIs
- do not invent attributes
- do not fabricate SDOH/clinical context
- allow different members to have different amounts of context
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.modeling.member_risk import (
    MemberRiskScoreAdapter,
)

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
    RankingCandidate,
)

from src.recommendations.recommendation_engine import (
    RecommendationEngine,
)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RISK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member_risk_scores.csv"
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
    / "validation"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "multi_member_pipeline_validation.json"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "multi_member_pipeline_validation.csv"
)


# ============================================================================
# HELPERS
# ============================================================================

def _get(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:

    if obj is None:
        return default

    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                value = obj[name]
                if value is not None:
                    return value
        return default

    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value

    return default


def _list(
    obj: Any,
    *names: str,
) -> list[Any]:

    value = _get(
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


def _serialize(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): _serialize(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _serialize(v)
            for v in value
        ]

    if hasattr(value, "to_dict"):
        return _serialize(
            value.to_dict()
        )

    if hasattr(
        value,
        "__dataclass_fields__",
    ):
        return _serialize(
            asdict(value)
        )

    if hasattr(value, "__dict__"):
        return {
            str(k): _serialize(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }

    return value


# ============================================================================
# GRAPH ID
# ============================================================================

def canonical_graph_member_id(
    member_id: str,
) -> str:

    value = str(member_id).strip()

    if value.startswith("member:"):
        return value

    return (
        "member:"
        + value.replace("-", "_")
    )


def resolve_graph_member_id(
    graph: dict[str, Any],
    member_id: str,
) -> str:

    canonical = canonical_graph_member_id(
        member_id
    )

    for node in graph.get(
        "nodes",
        [],
    ):

        if not isinstance(
            node,
            dict,
        ):
            continue

        node_id = str(
            node.get(
                "id",
                node.get(
                    "node_id",
                    "",
                ),
            )
        )

        if node_id == canonical:
            return canonical

    raise ValueError(
        "Member not found in graph: "
        f"{member_id}\n"
        f"Expected: {canonical}"
    )


# ============================================================================
# GRAPH
# ============================================================================

def load_validation_graph() -> dict[str, Any]:

    graph = load_graph(
        GRAPH_PATH
    )

    if not isinstance(
        graph,
        dict,
    ):
        raise TypeError(
            "Knowledge Graph must be a dictionary."
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

    return {
        **graph,
        "nodes": list(nodes),
        "relationships": list(
            relationships
        ),
    }


# ============================================================================
# MEMBER CONTEXT CLASSIFICATION
# ============================================================================

def classify_member(
    result: dict[str, Any],
) -> str:
    """
    Classify members based on actual observed context.

    This is only for validation grouping.
    It does not change pipeline behavior.
    """

    sdoh_count = result[
        "sdoh_factor_count"
    ]

    clinical_count = result[
        "clinical_factor_count"
    ]

    evidence_count = result[
        "evidence_count"
    ]

    if (
        sdoh_count >= 10
        and clinical_count >= 5
        and evidence_count >= 10
    ):
        return "rich_context"

    if (
        sdoh_count >= 1
        and clinical_count >= 1
    ):
        return "partial_context"

    if (
        sdoh_count >= 1
        and clinical_count == 0
    ):
        return "sdoh_only"

    if (
        sdoh_count == 0
        and clinical_count >= 1
    ):
        return "clinical_only"

    return "minimal_context"


# ============================================================================
# MEMBER PROCESSING
# ============================================================================

def process_member(
    member_id: str,
    adapter: MemberRiskScoreAdapter,
    graph: dict[str, Any],
    registry: Any,
) -> dict[str, Any]:

    record = adapter.require(
        member_id
    )

    risk_probability = (
        record.risk_probability
    )

    risk_band = (
        record.risk_band
    )

    graph_member_id = (
        resolve_graph_member_id(
            graph,
            member_id,
        )
    )

    reasoner = ContextualReasoner(
        graph=graph,
        registry=registry,
    )

    reasoning_result = (
        reasoner.reason(
            graph_member_id
        )
    )

    if reasoning_result is None:
        raise RuntimeError(
            "ContextualReasoner returned None."
        )

    # ------------------------------------------------------------------------
    # Actual member context
    # ------------------------------------------------------------------------

    sdoh_factors = _list(
        reasoning_result,
        "sdoh_factors",
    )

    sdoh_domains = _list(
        reasoning_result,
        "sdoh_domains",
    )

    clinical_factors = _list(
        reasoning_result,
        "clinical_factors",
    )

    evidence_records = _list(
        reasoning_result,
        "evidence_records",
    )

    intervention_candidates = _list(
        reasoning_result,
        "intervention_candidates",
    )

    # ------------------------------------------------------------------------
    # Risk consistency
    # ------------------------------------------------------------------------

    reasoning_probability = _get(
        reasoning_result,
        "risk_probability",
    )

    reasoning_band = _get(
        reasoning_result,
        "risk_band",
    )

    risk_probability_match = (
        reasoning_probability is None
        or abs(
            float(reasoning_probability)
            - float(risk_probability)
        ) < 1e-12
    )

    risk_band_match = (
        reasoning_band is None
        or str(reasoning_band)
        == str(risk_band)
    )

    if not risk_probability_match:
        raise AssertionError(
            f"Risk probability mismatch for {member_id}: "
            f"adapter={risk_probability}, "
            f"reasoning={reasoning_probability}"
        )

    if not risk_band_match:
        raise AssertionError(
            f"Risk band mismatch for {member_id}: "
            f"adapter={risk_band}, "
            f"reasoning={reasoning_band}"
        )

    # ------------------------------------------------------------------------
    # Prioritization
    # ------------------------------------------------------------------------

    prioritizer = (
        InterventionPrioritizer()
    )

    prioritization_result = (
        prioritizer.prioritize(
            reasoning_result
        )
    )

    priorities = _list(
        prioritization_result,
        "priorities",
    )

    # ------------------------------------------------------------------------
    # LambdaMART candidate preparation
    # ------------------------------------------------------------------------

    feature_builder = (
        RankingFeatureBuilder()
    )

    feature_records = (
        feature_builder.build_many(
            reasoning_result,
            priorities,
        )
    )

    if len(feature_records) != len(
        priorities
    ):
        raise AssertionError(
            "Feature-record count does not match "
            "prioritization count."
        )

    ranking_candidates = []

    for index, (
        priority,
        feature_record,
    ) in enumerate(
        zip(
            priorities,
            feature_records,
        ),
        start=1,
    ):

        matched_factors = _list(
            priority,
            "matched_factors",
        )

        evidence = _list(
            priority,
            "evidence",
        )

        intervention_id = str(
            _get(
                priority,
                "intervention_id",
                default="",
            )
        )

        if not intervention_id:
            raise AssertionError(
                f"Priority without intervention_id "
                f"for {member_id}"
            )

        feature_dict = {
            name: float(
                getattr(
                    feature_record,
                    name,
                )
            )
            for name in (
                "risk_probability",
                "risk_band_score",
                "sdoh_factor_count",
                "sdoh_domain_count",
                "clinical_factor_count",
                "evidence_count",
                "matched_factor_count",
                "factor_match_ratio",
                "intervention_context_score",
                "baseline_priority_score",
                "evidence_match_count",
                "evidence_density",
                "risk_x_matched_factors",
                "risk_x_context_score",
                "clinical_x_sdoh",
                "evidence_x_matched_factors",
                "candidate_rank",
            )
        }

        ranking_candidates.append(
            RankingCandidate(
                intervention_id=(
                    intervention_id
                ),
                intervention_name=str(
                    _get(
                        priority,
                        "name",
                        default=intervention_id,
                    )
                ),
                domain=str(
                    _get(
                        priority,
                        "domain",
                        default="",
                    )
                ),
                features=feature_dict,
                baseline_score=float(
                    _get(
                        priority,
                        "priority_score",
                        default=0.0,
                    )
                ),
                baseline_rank=int(
                    _get(
                        priority,
                        "rank",
                        default=index,
                    )
                ),
                matched_factor_count=len(
                    matched_factors
                ),
                evidence_match_count=len(
                    evidence
                ),
            )
        )

    # ------------------------------------------------------------------------
    # LambdaMART
    # ------------------------------------------------------------------------

    ranker = (
        LambdaMARTRanker()
    )

    ranked_result = (
        ranker.rank(
            ranking_candidates,
            member_id=graph_member_id,
        )
    )

    ranked_candidates = _list(
        ranked_result,
        "candidates",
    )

    if len(ranked_candidates) != len(
        ranking_candidates
    ):
        raise AssertionError(
            "LambdaMART result count does not "
            "match candidate count."
        )

    # ------------------------------------------------------------------------
    # Recommendation engine
    # ------------------------------------------------------------------------

    recommendation_engine = (
        RecommendationEngine()
    )

    recommendation_result = (
        recommendation_engine.generate(
            reasoning_result,
            prioritization_result,
        )
    )

    recommendations = _list(
        recommendation_result,
        "recommendations",
    )

    if len(recommendations) != len(
        priorities
    ):
        raise AssertionError(
            f"Recommendation count mismatch for {member_id}: "
            f"priorities={len(priorities)}, "
            f"recommendations={len(recommendations)}"
        )

    # ------------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------------

    result = {
        "member_id": member_id,
        "graph_member_id": graph_member_id,
        "risk_probability": risk_probability,
        "risk_band": risk_band,
        "risk_rank": record.risk_rank,
        "risk_percentile": record.risk_percentile,

        "sdoh_factor_count": len(
            sdoh_factors
        ),

        "sdoh_domain_count": len(
            sdoh_domains
        ),

        "clinical_factor_count": len(
            clinical_factors
        ),

        "evidence_count": len(
            evidence_records
        ),

        "intervention_candidate_count": len(
            intervention_candidates
        ),

        "prioritized_intervention_count": len(
            priorities
        ),

        "lambda_ranked_count": len(
            ranked_candidates
        ),

        "recommendation_count": len(
            recommendations
        ),

        "risk_probability_match": (
            risk_probability_match
        ),

        "risk_band_match": (
            risk_band_match
        ),

        "classification": None,

        "status": "PASS",

        "context": {
            "sdoh_factors": _serialize(
                sdoh_factors
            ),
            "sdoh_domains": _serialize(
                sdoh_domains
            ),
            "clinical_factors": _serialize(
                clinical_factors
            ),
            "evidence_records": _serialize(
                evidence_records
            ),
        },

        "recommendations": _serialize(
            recommendations
        ),
    }

    result["classification"] = (
        classify_member(result)
    )

    return result


# ============================================================================
# MEMBER SELECTION
# ============================================================================

def select_validation_members(
    adapter: MemberRiskScoreAdapter,
    graph: dict[str, Any],
    limit: int = 12,
) -> list[str]:
    """
    Select real members from the 108-member risk population.

    Selection is based on actual risk scores only. Context diversity is
    determined after running the pipeline; no fabricated metadata is used.
    """

    records = adapter.all_scores()

    # Highest-risk members
    highest = sorted(
        records,
        key=lambda item: (
            -item.risk_probability,
            item.risk_rank,
        ),
    )

    # Lowest-risk members
    lowest = sorted(
        records,
        key=lambda item: (
            item.risk_probability,
            item.risk_rank,
        ),
    )

    # Middle-risk members
    middle = sorted(
        records,
        key=lambda item: abs(
            item.risk_probability - 0.50
        ),
    )

    ordered = []

    def add_unique(record):
        if record.member_id in ordered:
            return

        try:
            resolve_graph_member_id(
                graph,
                record.member_id,
            )
        except ValueError:
            return

        ordered.append(
            record.member_id
        )

    # Mix risk levels.
    for record in highest[:4]:
        add_unique(record)

    for record in middle[:4]:
        add_unique(record)

    for record in lowest[:4]:
        add_unique(record)

    return ordered[:limit]


# ============================================================================
# VALIDATION REPORT
# ============================================================================

def write_reports(
    results: list[dict[str, Any]],
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "member_count": len(
            results
        ),
        "passed": sum(
            1
            for result in results
            if result["status"] == "PASS"
        ),
        "failed": sum(
            1
            for result in results
            if result["status"] == "FAIL"
        ),
        "classification_counts": {},
        "results": results,
    }

    for result in results:

        classification = result.get(
            "classification",
            "unknown",
        )

        summary[
            "classification_counts"
        ][classification] = (
            summary[
                "classification_counts"
            ].get(
                classification,
                0,
            )
            + 1
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            _serialize(summary),
            file,
            indent=2,
            ensure_ascii=False,
        )

    # CSV report
    import csv

    columns = [
        "member_id",
        "graph_member_id",
        "risk_probability",
        "risk_band",
        "risk_rank",
        "risk_percentile",
        "sdoh_factor_count",
        "sdoh_domain_count",
        "clinical_factor_count",
        "evidence_count",
        "intervention_candidate_count",
        "prioritized_intervention_count",
        "lambda_ranked_count",
        "recommendation_count",
        "risk_probability_match",
        "risk_band_match",
        "classification",
        "status",
    ]

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                {
                    column: result.get(
                        column,
                        "",
                    )
                    for column in columns
                }
            )


# ============================================================================
# SELF TEST
# ============================================================================

def run_validation() -> None:

    print("=" * 78)
    print(
        "HEALTHLENS MULTI-MEMBER PIPELINE VALIDATION"
    )
    print("=" * 78)

    # ------------------------------------------------------------------------
    # Risk adapter
    # ------------------------------------------------------------------------

    adapter = (
        MemberRiskScoreAdapter()
    )

    print(
        f"Risk records:                         "
        f"{adapter.member_count}"
    )

    if adapter.member_count <= 0:
        raise AssertionError(
            "No member risk records available."
        )

    # ------------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------------

    graph = load_validation_graph()

    print(
        f"Graph nodes:                          "
        f"{len(graph['nodes'])}"
    )

    print(
        f"Graph relationships:                  "
        f"{len(graph['relationships'])}"
    )

    # ------------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------------

    registry = build_registry()

    # ------------------------------------------------------------------------
    # Member selection
    # ------------------------------------------------------------------------

    selected_members = (
        select_validation_members(
            adapter,
            graph,
            limit=12,
        )
    )

    if not selected_members:
        raise AssertionError(
            "No real members were selected."
        )

    print()
    print(
        "Selected real members:"
    )

    for member_id in selected_members:
        print(
            f"  {member_id}"
        )

    # ------------------------------------------------------------------------
    # Process members
    # ------------------------------------------------------------------------

    results = []

    for index, member_id in enumerate(
        selected_members,
        start=1,
    ):

        print()
        print("-" * 78)

        print(
            f"[{index}/{len(selected_members)}] "
            f"{member_id}"
        )

        try:

            result = process_member(
                member_id=member_id,
                adapter=adapter,
                graph=graph,
                registry=registry,
            )

            results.append(
                result
            )

            print(
                f"Risk:                    "
                f"{result['risk_probability']:.6f}"
            )

            print(
                f"Risk band:               "
                f"{result['risk_band']}"
            )

            print(
                f"SDOH factors:            "
                f"{result['sdoh_factor_count']}"
            )

            print(
                f"SDOH domains:            "
                f"{result['sdoh_domain_count']}"
            )

            print(
                f"Clinical factors:        "
                f"{result['clinical_factor_count']}"
            )

            print(
                f"Evidence:                "
                f"{result['evidence_count']}"
            )

            print(
                f"Intervention candidates: "
                f"{result['intervention_candidate_count']}"
            )

            print(
                f"Recommendations:         "
                f"{result['recommendation_count']}"
            )

            print(
                f"Context class:           "
                f"{result['classification']}"
            )

            print(
                "Status:                  PASS"
            )

        except Exception as exc:

            failure = {
                "member_id": member_id,
                "status": "FAIL",
                "error_type": type(
                    exc
                ).__name__,
                "error": str(exc),
            }

            results.append(
                failure
            )

            print(
                f"Status:                  FAIL"
            )

            print(
                f"Error:                   "
                f"{type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------------

    write_reports(
        results
    )

    passed = sum(
        1
        for result in results
        if result.get(
            "status"
        ) == "PASS"
    )

    failed = sum(
        1
        for result in results
        if result.get(
            "status"
        ) == "FAIL"
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "MULTI-MEMBER VALIDATION RESULT"
    )
    print("=" * 78)

    print(
        f"Members tested:          "
        f"{len(results)}"
    )

    print(
        f"Passed:                  "
        f"{passed}"
    )

    print(
        f"Failed:                  "
        f"{failed}"
    )

    classifications = {}

    for result in results:

        classification = result.get(
            "classification"
        )

        if classification:

            classifications[
                classification
            ] = (
                classifications.get(
                    classification,
                    0,
                )
                + 1
            )

    print()
    print(
        "Context classifications:"
    )

    for name, count in sorted(
        classifications.items()
    ):

        print(
            f"  {name}: {count}"
        )

    print()
    print(
        "JSON output:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print(
        "CSV output:"
    )

    print(
        f"  {REPORT_PATH}"
    )

    print()

    if failed:
        raise AssertionError(
            f"Multi-member validation failed "
            f"for {failed} member(s)."
        )

    print("=" * 78)

    print(
        "MULTI-MEMBER PIPELINE VALIDATION: PASSED"
    )

    print("=" * 78)


if __name__ == "__main__":
    run_validation()