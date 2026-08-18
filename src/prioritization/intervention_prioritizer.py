"""
HealthLens Intervention Prioritization
=======================================

Converts a ContextualReasoner result into a deterministic,
explainable ranking of intervention candidates.

Architecture:

    ContextualReasoner
            |
            v
    ReasoningResult
            |
            v
    InterventionPrioritizer
            |
            v
    PrioritizationResult

IMPORTANT
---------
This layer does NOT:
    - recalculate risk
    - modify the Knowledge Graph
    - modify the Knowledge Base
    - generate new SDOH factors
    - replace evidence
    - train a new ML model

It only prioritizes interventions already identified by
the reasoning layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


# ============================================================================
# VERSION
# ============================================================================

PRIORITIZATION_VERSION = "1.0.0"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class InterventionPriority:
    """
    One ranked intervention recommendation.
    """

    intervention_id: str
    name: str
    domain: str

    matched_factors: list[str] = field(
        default_factory=list
    )

    base_score: float = 0.0

    priority_score: float = 0.0

    priority_band: str = "Low"

    rank: int = 0

    rationale: str = ""


@dataclass
class PrioritizationResult:
    """
    Complete prioritization result for one member.
    """

    member_id: str

    risk_probability: float | None = None

    risk_band: str | None = None

    priorities: list[InterventionPriority] = field(
        default_factory=list
    )

    reasoning_summary: dict[str, Any] = field(
        default_factory=dict
    )

    trace: list[str] = field(
        default_factory=list
    )

    version: str = PRIORITIZATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the prioritization result.
        """

        return asdict(self)


# ============================================================================
# PRIORITIZER
# ============================================================================

class InterventionPrioritizer:
    """
    Deterministic intervention prioritization engine.

    The prioritizer consumes the output of ContextualReasoner.

    It intentionally uses the intervention candidates already
    produced by the reasoning layer rather than independently
    discovering interventions.
    """

    def __init__(
        self,
        *,
        risk_weight: float = 0.0,
        factor_weight: float = 10.0,
        domain_weight: float = 1.0,
    ) -> None:

        self.risk_weight = float(
            risk_weight
        )

        self.factor_weight = float(
            factor_weight
        )

        self.domain_weight = float(
            domain_weight
        )

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def prioritize(
        self,
        reasoning_result: Any,
    ) -> PrioritizationResult:
        """
        Convert a ReasoningResult into ranked priorities.
        """

        member_id = str(
            getattr(
                reasoning_result,
                "member_id",
                "",
            )
        )

        risk_probability = (
            getattr(
                reasoning_result,
                "risk_probability",
                None,
            )
        )

        risk_band = (
            getattr(
                reasoning_result,
                "risk_band",
                None,
            )
        )

        candidates = (
            getattr(
                reasoning_result,
                "intervention_candidates",
                [],
            )
            or []
        )

        trace: list[str] = []

        trace.append(
            "Prioritization started."
        )

        trace.append(
            f"Member: {member_id}"
        )

        trace.append(
            f"Intervention candidates received: "
            f"{len(candidates)}"
        )

        priorities: list[
            InterventionPriority
        ] = []

        for candidate in candidates:

            priority = (
                self._score_candidate(
                    candidate=candidate,
                    risk_probability=(
                        risk_probability
                    ),
                    risk_band=risk_band,
                )
            )

            priorities.append(
                priority
            )

        # --------------------------------------------------------------------
        # Deterministic ranking
        # --------------------------------------------------------------------

        priorities.sort(
            key=lambda item: (
                -item.priority_score,
                -len(item.matched_factors),
                item.intervention_id,
            )
        )

        # --------------------------------------------------------------------
        # Assign ranks
        # --------------------------------------------------------------------

        ranked: list[
            InterventionPriority
        ] = []

        for index, priority in enumerate(
            priorities,
            start=1,
        ):

            ranked.append(
                InterventionPriority(
                    intervention_id=(
                        priority.intervention_id
                    ),
                    name=priority.name,
                    domain=priority.domain,
                    matched_factors=list(
                        priority.matched_factors
                    ),
                    base_score=(
                        priority.base_score
                    ),
                    priority_score=(
                        priority.priority_score
                    ),
                    priority_band=(
                        priority.priority_band
                    ),
                    rank=index,
                    rationale=(
                        priority.rationale
                    ),
                )
            )

        priorities = ranked

        trace.append(
            f"Interventions ranked: "
            f"{len(priorities)}"
        )

        if priorities:

            trace.append(
                "Top intervention: "
                f"{priorities[0].intervention_id}"
            )

        else:

            trace.append(
                "No intervention candidates "
                "were available."
            )

        trace.append(
            "Prioritization completed."
        )

        # --------------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------------

        domains = sorted(
            {
                priority.domain
                for priority in priorities
                if priority.domain
            }
        )

        factors = sorted(
            {
                factor
                for priority in priorities
                for factor in (
                    priority.matched_factors
                )
            }
        )

        summary = {
            "intervention_count": len(
                priorities
            ),
            "domain_count": len(
                domains
            ),
            "factor_count": len(
                factors
            ),
            "domains": domains,
            "factors": factors,
            "top_intervention": (
                priorities[0].intervention_id
                if priorities
                else None
            ),
        }

        return PrioritizationResult(
            member_id=member_id,
            risk_probability=(
                self._safe_float(
                    risk_probability
                )
            ),
            risk_band=(
                str(risk_band)
                if risk_band is not None
                else None
            ),
            priorities=priorities,
            reasoning_summary=summary,
            trace=trace,
        )

    # ========================================================================
    # SCORING
    # ========================================================================

    def _score_candidate(
        self,
        *,
        candidate: Any,
        risk_probability: float | None,
        risk_band: str | None,
    ) -> InterventionPriority:
        """
        Score one intervention candidate.

        Existing candidate score is treated as the primary
        contextual score.

        Risk is currently metadata only by default
        (risk_weight = 0.0), preventing the very high member
        risk score from overwhelming contextual matching.
        """

        intervention_id = self._get_value(
            candidate,
            "intervention_id",
            "id",
        )

        name = self._get_value(
            candidate,
            "name",
            "intervention_name",
            "title",
        )

        domain = self._get_value(
            candidate,
            "domain",
        )

        matched_factors = (
            self._get_list(
                candidate,
                "matched_factors",
                "factors",
                "target_factors",
            )
        )

        base_score = self._safe_float(
            self._get_value(
                candidate,
                "score",
                "base_score",
                "match_score",
            )
        )

        # --------------------------------------------------------------------
        # Contextual factor contribution
        # --------------------------------------------------------------------

        factor_component = (
            len(matched_factors)
            * self.factor_weight
        )

        # --------------------------------------------------------------------
        # Domain contribution
        # --------------------------------------------------------------------

        domain_component = (
            self.domain_weight
            if domain
            else 0.0
        )

        # --------------------------------------------------------------------
        # Risk contribution
        # --------------------------------------------------------------------

        risk_component = 0.0

        if (
            risk_probability is not None
            and self.risk_weight != 0.0
        ):
            risk_component = (
                risk_probability
                * self.risk_weight
            )

        priority_score = (
            base_score
            + factor_component
            + domain_component
            + risk_component
        )

        priority_band = (
            self._priority_band(
                priority_score
            )
        )

        rationale = (
            self._build_rationale(
                name=name,
                domain=domain,
                matched_factors=(
                    matched_factors
                ),
                base_score=base_score,
                priority_score=(
                    priority_score
                ),
                risk_probability=(
                    risk_probability
                ),
                risk_band=risk_band,
            )
        )

        return InterventionPriority(
            intervention_id=(
                intervention_id
            ),
            name=name,
            domain=domain,
            matched_factors=(
                matched_factors
            ),
            base_score=base_score,
            priority_score=priority_score,
            priority_band=priority_band,
            rationale=rationale,
        )

    # ========================================================================
    # PRIORITY BAND
    # ========================================================================

    @staticmethod
    def _priority_band(
        score: float,
    ) -> str:
        """
        Convert priority score into a human-readable band.

        Thresholds are intentionally simple and deterministic.
        """

        if score >= 50:
            return "Critical"

        if score >= 30:
            return "High"

        if score >= 15:
            return "Moderate"

        return "Low"

    # ========================================================================
    # RATIONALE
    # ========================================================================

    @staticmethod
    def _build_rationale(
        *,
        name: str,
        domain: str,
        matched_factors: Sequence[str],
        base_score: float,
        priority_score: float,
        risk_probability: float | None,
        risk_band: str | None,
    ) -> str:
        """
        Build an explainable, deterministic rationale.
        """

        factor_count = len(
            matched_factors
        )

        parts = [
            f"{name} was prioritized",
            f"for the {domain} domain",
            f"with {factor_count} matched SDOH factor"
            f"{'s' if factor_count != 1 else ''}",
            f"(context score {base_score:g})",
        ]

        if (
            risk_probability is not None
            and risk_band
        ):
            parts.append(
                f"for a member with "
                f"{risk_band.lower()} overall risk"
            )

        parts.append(
            f"(priority score {priority_score:g})"
        )

        return " ".join(parts) + "."

    # ========================================================================
    # VALUE HELPERS
    # ========================================================================

    @staticmethod
    def _get_value(
        obj: Any,
        *names: str,
    ) -> str:
        """
        Read a value from either a mapping or object.
        """

        for name in names:

            if isinstance(
                obj,
                Mapping,
            ):

                value = obj.get(
                    name
                )

            else:

                value = getattr(
                    obj,
                    name,
                    None,
                )

            if value is not None:

                return str(
                    value
                ).strip()

        return ""

    @staticmethod
    def _get_list(
        obj: Any,
        *names: str,
    ) -> list[str]:
        """
        Read a list-like value from either a mapping or object.
        """

        for name in names:

            if isinstance(
                obj,
                Mapping,
            ):

                value = obj.get(
                    name
                )

            else:

                value = getattr(
                    obj,
                    name,
                    None,
                )

            if value is None:
                continue

            if isinstance(
                value,
                str,
            ):

                value = [
                    value
                ]

            try:

                return [
                    str(item)
                    for item in value
                    if str(item).strip()
                ]

            except TypeError:

                return []

        return []

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:
        """
        Safely convert a value to float.
        """

        try:

            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def prioritize_member(
    reasoning_result: Any,
) -> PrioritizationResult:
    """
    Convenience wrapper around InterventionPrioritizer.
    """

    prioritizer = (
        InterventionPrioritizer()
    )

    return prioritizer.prioritize(
        reasoning_result
    )


# ============================================================================
# SELF TEST
# ============================================================================

def _self_test() -> None:
    """
    Basic standalone prioritization self-test.

    Uses a small synthetic reasoning result so this module
    does not modify or depend on the Knowledge Graph.
    """

    print("=" * 70)
    print(
        "HEALTHLENS INTERVENTION PRIORITIZATION"
    )
    print("=" * 70)

    # ------------------------------------------------------------------
    # Lightweight test object
    # ------------------------------------------------------------------

    class TestReasoningResult:

        member_id = (
            "test-member"
        )

        risk_probability = (
            0.9999
        )

        risk_band = (
            "Very High"
        )

        intervention_candidates = [
            {
                "intervention_id":
                    "INT_HOUSING_STABILITY",
                "name":
                    "Housing stability and housing-support referral",
                "domain":
                    "Housing",
                "matched_factors": [
                    "housing_rent_35_plus_pct",
                    "housing_crowded_1_01_to_1_50_pct",
                    "housing_crowded_1_51_plus_pct",
                    "housing_rent_30_to_34_9_pct",
                ],
                "score":
                    40,
            },
            {
                "intervention_id":
                    "INT_TRANSPORTATION",
                "name":
                    "Transportation assistance for healthcare access",
                "domain":
                    "Transportation",
                "matched_factors": [
                    "straight_no_vehicle_households_beyond_1mi_count_sum",
                    "driving_low_income_low_access_tract_count",
                    "driving_no_vehicle_households_beyond_1mi_count_sum",
                    "driving_low_access_population_beyond_1mi_10mi_count_sum",
                ],
                "score":
                    40,
            },
            {
                "intervention_id":
                    "INT_ECONOMIC_BENEFITS",
                "name":
                    "Financial assistance and benefits navigation",
                "domain":
                    "Economic Stability",
                "matched_factors": [
                    "snap_households_count_sum",
                ],
                "score":
                    10,
            },
        ]

    # ------------------------------------------------------------------
    # Build prioritizer
    # ------------------------------------------------------------------

    prioritizer = (
        InterventionPrioritizer()
    )

    print(
        "Prioritizer construction: PASS"
    )

    result = (
        prioritizer.prioritize(
            TestReasoningResult()
        )
    )

    assert result.member_id == (
        "test-member"
    )

    print(
        "Prioritization execution: PASS"
    )

    assert result.risk_probability == (
        0.9999
    )

    print(
        "Risk propagation:          PASS"
    )

    assert len(
        result.priorities
    ) == 3

    print(
        "Candidate processing:      PASS"
    )

    assert (
        result.priorities[0].rank
        == 1
    )

    print(
        "Ranking:                   PASS"
    )

    assert all(
        priority.priority_score
        >= 0
        for priority in result.priorities
    )

    print(
        "Score validation:          PASS"
    )

    assert all(
        priority.rationale
        for priority in result.priorities
    )

    print(
        "Explainability:            PASS"
    )

    serialized = (
        result.to_dict()
    )

    assert isinstance(
        serialized,
        dict,
    )

    assert isinstance(
        serialized[
            "priorities"
        ],
        list,
    )

    print(
        "Serialization:             PASS"
    )

    print()
    print(
        "PRIORITIZATION RESULT"
    )
    print("=" * 70)

    for priority in result.priorities:

        print(
            f"{priority.rank}. "
            f"{priority.intervention_id} | "
            f"{priority.name} | "
            f"score: "
            f"{priority.priority_score:g} | "
            f"{priority.priority_band}"
        )

    print()
    print(
        "INTERVENTION PRIORITIZATION "
        "SELF-TEST: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    _self_test()