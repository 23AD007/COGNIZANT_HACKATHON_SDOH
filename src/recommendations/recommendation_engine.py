"""
HealthLens Recommendation Engine
================================

Converts intervention prioritization results into
actionable, explainable care recommendations.

Pipeline:

    ReasoningResult
          |
          v
    PrioritizationResult
          |
          v
    RecommendationEngine
          |
          v
    RecommendationResult

Design principles:
    - deterministic
    - explainable
    - evidence-aware
    - no additional ML model
    - preserves reasoning/prioritization outputs
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
)
from typing import Any


# ============================================================================
# VERSION
# ============================================================================

RECOMMENDATION_ENGINE_VERSION = "1.0.0"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class Recommendation:
    """
    One actionable recommendation generated from
    an intervention priority.
    """

    rank: int

    intervention_id: str

    intervention_name: str

    domain: str

    priority_score: float

    priority_band: str

    urgency: str

    matched_factors: list[str] = field(
        default_factory=list
    )

    evidence_count: int = 0

    evidence_ids: list[str] = field(
        default_factory=list
    )

    action: str = ""

    rationale: str = ""

    supporting_factors: list[str] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class RecommendationResult:
    """
    Complete recommendation result for one member.
    """

    member_id: str

    risk_probability: float | None

    risk_band: str | None

    recommendations: list[Recommendation] = field(
        default_factory=list
    )

    total_recommendations: int = 0

    engine_version: str = (
        RECOMMENDATION_ENGINE_VERSION
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the recommendation result.
        """

        return asdict(self)


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def _get_attr(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve an attribute from an object.

    Also supports dictionaries so that the engine can
    consume either dataclass objects or serialized data.
    """

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
    Retrieve a collection safely.
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


def _string_list(
    values: Any,
) -> list[str]:
    """
    Convert a collection to a list of strings.
    """

    if values is None:
        return []

    if not isinstance(
        values,
        (list, tuple, set),
    ):
        values = [values]

    result = []

    for value in values:

        if isinstance(
            value,
            str,
        ):

            result.append(
                value
            )

        elif isinstance(
            value,
            dict,
        ):

            factor = (
                value.get(
                    "factor"
                )
                or value.get(
                    "id"
                )
                or value.get(
                    "name"
                )
            )

            if factor:
                result.append(
                    str(factor)
                )

        else:

            factor = (
                getattr(
                    value,
                    "factor",
                    None,
                )
                or getattr(
                    value,
                    "id",
                    None,
                )
                or getattr(
                    value,
                    "name",
                    None,
                )
            )

            if factor:
                result.append(
                    str(factor)
                )

    return result


# ============================================================================
# PRIORITY EXTRACTION
# ============================================================================

def _extract_priorities(
    prioritization_result: Any,
) -> list[Any]:
    """
    Extract ranked intervention priorities.
    """

    priorities = _collection(
        prioritization_result,
        "priorities",
        "ranked_interventions",
        "interventions",
    )

    return priorities


# ============================================================================
# EVIDENCE EXTRACTION
# ============================================================================

def _extract_evidence_for_factors(
    reasoning_result: Any,
    factors: list[str],
) -> list[Any]:
    """
    Resolve evidence records associated with matched
    SDOH factors.

    Evidence is taken from the reasoning result.
    """

    evidence_records = _collection(
        reasoning_result,
        "evidence_records",
        "evidence",
    )

    if not evidence_records:
        return []

    factor_set = set(
        factors
    )

    matched = []

    for evidence in evidence_records:

        evidence_factor = _get_attr(
            evidence,
            "factor",
            "sdoh_factor",
            default=None,
        )

        if (
            evidence_factor is not None
            and str(evidence_factor)
            in factor_set
        ):

            matched.append(
                evidence
            )

    return matched


# ============================================================================
# EVIDENCE IDS
# ============================================================================

def _evidence_ids(
    evidence_records: list[Any],
) -> list[str]:
    """
    Extract evidence IDs where available.
    """

    result = []

    for evidence in evidence_records:

        evidence_id = _get_attr(
            evidence,
            "evidence_id",
            "id",
            default=None,
        )

        if evidence_id:

            result.append(
                str(evidence_id)
            )

    return result


# ============================================================================
# URGENCY
# ============================================================================

def _determine_urgency(
    priority_band: str,
    risk_band: str | None,
) -> str:
    """
    Determine recommendation urgency.

    Priority band takes precedence because it directly
    represents intervention priority.
    """

    priority = str(
        priority_band or ""
    ).strip().lower()

    risk = str(
        risk_band or ""
    ).strip().lower()

    if priority == "critical":

        return "Immediate"

    if priority == "high":

        return "High"

    if priority == "moderate":

        if risk in {
            "very high",
            "high",
        }:
            return "High"

        return "Moderate"

    if priority == "low":

        return "Routine"

    if risk == "very high":

        return "Immediate"

    if risk == "high":

        return "High"

    return "Routine"


# ============================================================================
# ACTION GENERATION
# ============================================================================

def _build_action(
    intervention_name: str,
    domain: str,
    matched_factors: list[str],
) -> str:
    """
    Build an actionable but non-clinical recommendation.

    The recommendation describes the intervention to
    consider; it does not prescribe medical treatment.
    """

    factor_count = len(
        matched_factors
    )

    if factor_count == 1:

        factor_text = (
            "1 identified SDOH factor"
        )

    else:

        factor_text = (
            f"{factor_count} identified SDOH factors"
        )

    return (
        f"Consider {intervention_name} "
        f"for the {domain} domain, "
        f"addressing {factor_text}."
    )


# ============================================================================
# RATIONALE
# ============================================================================

def _build_rationale(
    intervention_name: str,
    domain: str,
    matched_factors: list[str],
    priority_score: float,
    priority_band: str,
    risk_band: str | None,
    evidence_count: int,
) -> str:
    """
    Build a transparent explanation.
    """

    factor_count = len(
        matched_factors
    )

    risk_text = (
        str(risk_band)
        if risk_band
        else "unknown"
    )

    if evidence_count == 1:

        evidence_text = (
            "1 supporting evidence record"
        )

    else:

        evidence_text = (
            f"{evidence_count} supporting evidence records"
        )

    return (
        f"{intervention_name} is recommended for "
        f"the {domain} domain because "
        f"{factor_count} SDOH factor"
        f"{'s' if factor_count != 1 else ''} "
        f"matched the intervention. "
        f"The intervention has a priority score of "
        f"{priority_score:g} and a {priority_band} "
        f"priority level for a member with "
        f"{risk_text} overall risk. "
        f"The recommendation is supported by "
        f"{evidence_text}."
    )


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class RecommendationEngine:
    """
    Deterministic recommendation engine.

    Input:
        ReasoningResult
        PrioritizationResult

    Output:
        RecommendationResult
    """

    def __init__(
        self,
        *,
        version: str = RECOMMENDATION_ENGINE_VERSION,
    ) -> None:

        self.version = str(
            version
        )

    # ------------------------------------------------------------------------
    # BUILD RECOMMENDATION
    # ------------------------------------------------------------------------

    def _build_recommendation(
        self,
        priority: Any,
        reasoning_result: Any,
    ) -> Recommendation:

        rank = int(
            _get_attr(
                priority,
                "rank",
                default=0,
            )
        )

        intervention_id = str(
            _get_attr(
                priority,
                "intervention_id",
                "id",
                default="",
            )
        )

        intervention_name = str(
            _get_attr(
                priority,
                "name",
                "intervention_name",
                "description",
                default=intervention_id,
            )
        )

        domain = str(
            _get_attr(
                priority,
                "domain",
                default="",
            )
        )

        priority_score_raw = _get_attr(
            priority,
            "priority_score",
            "score",
            default=0.0,
        )

        priority_score = float(
            priority_score_raw
        )

        priority_band = str(
            _get_attr(
                priority,
                "priority_band",
                "band",
                "severity",
                default="Unknown",
            )
        )

        matched_factors = _string_list(
            _get_attr(
                priority,
                "matched_factors",
                "factors",
                default=[],
            )
        )

        risk_band = _get_attr(
            reasoning_result,
            "risk_band",
            "risk_level",
            default=None,
        )

        evidence_records = (
            _extract_evidence_for_factors(
                reasoning_result,
                matched_factors,
            )
        )

        evidence_ids = _evidence_ids(
            evidence_records
        )

        urgency = _determine_urgency(
            priority_band,
            risk_band,
        )

        action = _build_action(
            intervention_name,
            domain,
            matched_factors,
        )

        rationale = _build_rationale(
            intervention_name,
            domain,
            matched_factors,
            priority_score,
            priority_band,
            risk_band,
            len(evidence_records),
        )

        return Recommendation(
            rank=rank,
            intervention_id=intervention_id,
            intervention_name=intervention_name,
            domain=domain,
            priority_score=priority_score,
            priority_band=priority_band,
            urgency=urgency,
            matched_factors=matched_factors,
            evidence_count=len(
                evidence_records
            ),
            evidence_ids=evidence_ids,
            action=action,
            rationale=rationale,
            supporting_factors=matched_factors.copy(),
        )

    # ------------------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------------------

    def generate(
        self,
        reasoning_result: Any,
        prioritization_result: Any,
    ) -> RecommendationResult:
        """
        Generate recommendations from reasoning and
        prioritization results.
        """

        if reasoning_result is None:

            raise ValueError(
                "reasoning_result cannot be None."
            )

        if prioritization_result is None:

            raise ValueError(
                "prioritization_result cannot be None."
            )

        member_id = str(
            _get_attr(
                reasoning_result,
                "member_id",
                default="",
            )
        )

        if not member_id:

            raise ValueError(
                "Reasoning result has no member_id."
            )

        risk_probability = _get_attr(
            reasoning_result,
            "risk_probability",
            "risk_score",
            default=None,
        )

        if risk_probability is not None:

            risk_probability = float(
                risk_probability
            )

        risk_band = _get_attr(
            reasoning_result,
            "risk_band",
            "risk_level",
            default=None,
        )

        if risk_band is not None:

            risk_band = str(
                risk_band
            )

        priorities = _extract_priorities(
            prioritization_result
        )

        recommendations = []

        for priority in priorities:

            recommendation = (
                self._build_recommendation(
                    priority,
                    reasoning_result,
                )
            )

            recommendations.append(
                recommendation
            )

        # Ensure rank order.
        recommendations.sort(
            key=lambda item: (
                item.rank,
                -item.priority_score,
            )
        )

        # Reassign ranks if the source prioritizer
        # returned valid ordering but missing ranks.
        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):

            if recommendation.rank <= 0:

                recommendations[index - 1] = (
                    Recommendation(
                        **{
                            **asdict(
                                recommendation
                            ),
                            "rank": index,
                        }
                    )
                )

        return RecommendationResult(
            member_id=member_id,
            risk_probability=risk_probability,
            risk_band=risk_band,
            recommendations=recommendations,
            total_recommendations=len(
                recommendations
            ),
            engine_version=self.version,
        )

    # ------------------------------------------------------------------------
    # ALIAS
    # ------------------------------------------------------------------------

    def recommend(
        self,
        reasoning_result: Any,
        prioritization_result: Any,
    ) -> RecommendationResult:
        """
        Public alias for generate().
        """

        return self.generate(
            reasoning_result,
            prioritization_result,
        )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_recommendation_result(
    result: RecommendationResult,
) -> None:
    """
    Validate recommendation output.
    """

    if not result.member_id:

        raise AssertionError(
            "Recommendation result has no member ID."
        )

    if result.risk_probability is not None:

        if not (
            0.0
            <= result.risk_probability
            <= 1.0
        ):

            raise AssertionError(
                "Risk probability must be between 0 and 1."
            )

    if result.total_recommendations != len(
        result.recommendations
    ):

        raise AssertionError(
            "Recommendation count mismatch."
        )

    previous_rank = 0

    for recommendation in result.recommendations:

        if not recommendation.intervention_id:

            raise AssertionError(
                "Recommendation has no intervention ID."
            )

        if not recommendation.intervention_name:

            raise AssertionError(
                "Recommendation has no intervention name."
            )

        if not recommendation.domain:

            raise AssertionError(
                "Recommendation has no domain."
            )

        if recommendation.priority_score < 0:

            raise AssertionError(
                "Recommendation has negative priority score."
            )

        if not recommendation.action:

            raise AssertionError(
                "Recommendation has no action."
            )

        if not recommendation.rationale:

            raise AssertionError(
                "Recommendation has no rationale."
            )

        if recommendation.rank < previous_rank:

            raise AssertionError(
                "Recommendations are not ordered by rank."
            )

        previous_rank = recommendation.rank


# ============================================================================
# SERIALIZATION
# ============================================================================

def serialize_recommendations(
    result: RecommendationResult,
) -> dict[str, Any]:
    """
    Serialize recommendations.
    """

    validate_recommendation_result(
        result
    )

    return result.to_dict()


# ============================================================================
# SELF TEST
# ============================================================================

def _self_test() -> None:

    print("=" * 70)

    print(
        "HEALTHLENS RECOMMENDATION ENGINE"
    )

    print("=" * 70)

    # ------------------------------------------------------------------------
    # Mock-like objects used ONLY for unit testing the recommendation
    # transformation itself.
    #
    # Production integration will use the real ReasoningResult and
    # PrioritizationResult.
    # ------------------------------------------------------------------------

    class Evidence:

        def __init__(
            self,
            evidence_id: str,
            factor: str,
        ):

            self.evidence_id = evidence_id
            self.factor = factor

    class Reasoning:

        member_id = (
            "member:test_member"
        )

        risk_probability = 0.95

        risk_band = "Very High"

        evidence_records = [
            Evidence(
                "evidence_001",
                "housing_rent_35_plus_pct",
            ),
            Evidence(
                "evidence_002",
                "housing_crowded_1_01_to_1_50_pct",
            ),
        ]

    class Priority:

        rank = 1

        intervention_id = (
            "INT_HOUSING_STABILITY"
        )

        name = (
            "Housing stability and housing-support referral"
        )

        domain = "Housing"

        priority_score = 82.0

        priority_band = "Critical"

        matched_factors = [
            "housing_rent_35_plus_pct",
            "housing_crowded_1_01_to_1_50_pct",
        ]

    class Prioritization:

        priorities = [
            Priority()
        ]

    reasoning = Reasoning()

    prioritization = Prioritization()

    # ------------------------------------------------------------------------
    # ENGINE
    # ------------------------------------------------------------------------

    engine = RecommendationEngine()

    print(
        "Recommendation engine construction: PASS"
    )

    # ------------------------------------------------------------------------
    # GENERATE
    # ------------------------------------------------------------------------

    result = engine.generate(
        reasoning,
        prioritization,
    )

    print(
        "Recommendation generation:            PASS"
    )

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    validate_recommendation_result(
        result
    )

    print(
        "Recommendation validation:             PASS"
    )

    # ------------------------------------------------------------------------
    # COUNT
    # ------------------------------------------------------------------------

    assert (
        result.total_recommendations
        == 1
    )

    print(
        "Recommendation count:                  PASS"
    )

    # ------------------------------------------------------------------------
    # EVIDENCE
    # ------------------------------------------------------------------------

    recommendation = (
        result.recommendations[0]
    )

    assert (
        recommendation.evidence_count
        == 2
    )

    assert len(
        recommendation.evidence_ids
    ) == 2

    print(
        "Evidence linkage:                      PASS"
    )

    # ------------------------------------------------------------------------
    # FACTORS
    # ------------------------------------------------------------------------

    assert len(
        recommendation.matched_factors
    ) == 2

    print(
        "Factor linkage:                        PASS"
    )

    # ------------------------------------------------------------------------
    # URGENCY
    # ------------------------------------------------------------------------

    assert (
        recommendation.urgency
        == "Immediate"
    )

    print(
        "Urgency calculation:                   PASS"
    )

    # ------------------------------------------------------------------------
    # ACTION
    # ------------------------------------------------------------------------

    assert recommendation.action

    print(
        "Action generation:                     PASS"
    )

    # ------------------------------------------------------------------------
    # RATIONALE
    # ------------------------------------------------------------------------

    assert recommendation.rationale

    print(
        "Explainability:                        PASS"
    )

    # ------------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------------

    serialized = (
        serialize_recommendations(
            result
        )
    )

    assert isinstance(
        serialized,
        dict,
    )

    assert (
        serialized["member_id"]
        == "member:test_member"
    )

    assert (
        len(
            serialized[
                "recommendations"
            ]
        )
        == 1
    )

    print(
        "Serialization:                        PASS"
    )

    # ------------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------------

    print()

    print(
        "RECOMMENDATION ENGINE SELF-TEST: PASSED"
    )

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    _self_test()