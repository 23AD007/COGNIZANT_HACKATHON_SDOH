"""
HealthLens — LambdaMART Ranking Feature Construction
=====================================================

Purpose
-------
Build feature vectors for candidate intervention ranking.

Pipeline:

    ContextualReasoner
            ↓
    Intervention candidates
            ↓
    RankingFeatureBuilder
            ↓
    LambdaMART-ready feature matrix
            ↓
    LambdaMART training / inference

Important
---------
This module DOES NOT train LambdaMART.

It only constructs stable, explainable ranking features.

The current rule-based InterventionPrioritizer remains the
baseline until a properly labelled intervention-outcome
training dataset is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json


# ============================================================================
# VERSION
# ============================================================================

FEATURE_SCHEMA_VERSION = "1.0.0"


# ============================================================================
# FEATURE RECORD
# ============================================================================

@dataclass(frozen=True)
class InterventionRankingFeatures:
    """
    Feature vector for one member/intervention pair.

    All numerical features are explicitly represented so that
    the output can later be passed directly to a LambdaMART
    implementation.
    """

    member_id: str
    intervention_id: str

    # ------------------------------------------------------------------
    # Member risk/context
    # ------------------------------------------------------------------

    risk_probability: float
    risk_band_score: float

    sdoh_factor_count: float
    sdoh_domain_count: float
    clinical_factor_count: float
    evidence_count: float

    # ------------------------------------------------------------------
    # Intervention matching
    # ------------------------------------------------------------------

    matched_factor_count: float
    factor_match_ratio: float

    intervention_context_score: float
    baseline_priority_score: float

    # ------------------------------------------------------------------
    # Evidence/context strength
    # ------------------------------------------------------------------

    evidence_match_count: float
    evidence_density: float

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------

    risk_x_matched_factors: float
    risk_x_context_score: float
    clinical_x_sdoh: float
    evidence_x_matched_factors: float

    # ------------------------------------------------------------------
    # Candidate rank metadata
    # ------------------------------------------------------------------

    candidate_rank: float


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def _get(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """
    Retrieve a value from either an object or mapping.
    """

    if obj is None:
        return default

    if isinstance(
        obj,
        Mapping,
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


def _list(
    obj: Any,
    *names: str,
) -> list[Any]:
    """
    Retrieve a collection safely.
    """

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


def _float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safe numeric conversion.
    """

    if value is None:
        return default

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _normalise(
    value: Any,
) -> str:
    """
    Normalize identifiers.
    """

    return (
        str(value)
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
    )


# ============================================================================
# RISK BAND ENCODING
# ============================================================================

def risk_band_score(
    risk_band: Any,
) -> float:
    """
    Convert the existing risk band into an ordinal feature.

    The mapping is deliberately simple and deterministic.
    """

    normalized = _normalise(
        risk_band
    )

    mapping = {
        "low": 1.0,
        "moderate": 2.0,
        "medium": 2.0,
        "high": 3.0,
        "very_high": 4.0,
        "critical": 5.0,
    }

    return mapping.get(
        normalized,
        0.0,
    )


# ============================================================================
# MEMBER CONTEXT
# ============================================================================

@dataclass(frozen=True)
class MemberRankingContext:
    """
    Common features shared by all candidate interventions
    for a member.
    """

    member_id: str

    risk_probability: float
    risk_band_score: float

    sdoh_factor_count: float
    sdoh_domain_count: float
    clinical_factor_count: float
    evidence_count: float


def build_member_context(
    reasoning_result: Any,
) -> MemberRankingContext:
    """
    Construct the member-level ranking context from the
    existing ContextualReasoner output.
    """

    member_id = _get(
        reasoning_result,
        "member_id",
        default="",
    )

    risk_probability = _float(
        _get(
            reasoning_result,
            "risk_probability",
            "risk_score",
            default=0.0,
        )
    )

    risk_band = _get(
        reasoning_result,
        "risk_band",
        "risk_level",
        default="",
    )

    sdoh_factors = _list(
        reasoning_result,
        "sdoh_factors",
        "relevant_sdoh_factors",
    )

    domains = _list(
        reasoning_result,
        "sdoh_domains",
        "domains",
    )

    clinical_factors = _list(
        reasoning_result,
        "clinical_factors",
        "clinical_context",
    )

    evidence = _list(
        reasoning_result,
        "evidence_records",
        "evidence",
    )

    return MemberRankingContext(
        member_id=str(
            member_id
        ),

        risk_probability=risk_probability,

        risk_band_score=risk_band_score(
            risk_band
        ),

        sdoh_factor_count=float(
            len(sdoh_factors)
        ),

        sdoh_domain_count=float(
            len(domains)
        ),

        clinical_factor_count=float(
            len(clinical_factors)
        ),

        evidence_count=float(
            len(evidence)
        ),
    )


# ============================================================================
# INTERVENTION FEATURE EXTRACTION
# ============================================================================

def _matched_factors(
    candidate: Any,
) -> list[Any]:
    """
    Extract matched factors from an intervention candidate.
    """

    return _list(
        candidate,
        "matched_factors",
        "matching_factors",
        "factors",
    )


def _candidate_score(
    candidate: Any,
) -> float:
    """
    Extract the existing contextual/baseline score.
    """

    return _float(
        _get(
            candidate,
            "score",
            "context_score",
            "priority_score",
            "baseline_priority_score",
            default=0.0,
        )
    )


def _candidate_id(
    candidate: Any,
) -> str:
    """
    Extract intervention ID.
    """

    return str(
        _get(
            candidate,
            "intervention_id",
            "id",
            default="",
        )
    )


def _candidate_evidence(
    candidate: Any,
) -> list[Any]:
    """
    Extract candidate-specific evidence where available.
    """

    return _list(
        candidate,
        "evidence",
        "evidence_records",
        "supporting_evidence",
    )


# ============================================================================
# FEATURE BUILDER
# ============================================================================

class RankingFeatureBuilder:
    """
    Builds LambdaMART-ready feature records.
    """

    def __init__(
        self,
    ) -> None:

        self.feature_schema_version = (
            FEATURE_SCHEMA_VERSION
        )

    # ------------------------------------------------------------------
    # Single candidate
    # ------------------------------------------------------------------

    def build(
        self,
        reasoning_result: Any,
        candidate: Any,
        candidate_rank: int = 0,
    ) -> InterventionRankingFeatures:
        """
        Build one member/intervention feature vector.
        """

        context = build_member_context(
            reasoning_result
        )

        matched_factors = _matched_factors(
            candidate
        )

        matched_count = float(
            len(
                matched_factors
            )
        )

        candidate_score = _candidate_score(
            candidate
        )

        candidate_evidence = _candidate_evidence(
            candidate
        )

        # --------------------------------------------------------------
        # Factor match ratio
        # --------------------------------------------------------------

        if (
            context.sdoh_factor_count
            > 0
        ):

            factor_match_ratio = (
                matched_count
                / context.sdoh_factor_count
            )

        else:

            factor_match_ratio = 0.0

        # --------------------------------------------------------------
        # Evidence density
        # --------------------------------------------------------------

        if (
            matched_count
            > 0
        ):

            evidence_density = (
                float(
                    len(
                        candidate_evidence
                    )
                )
                / matched_count
            )

        else:

            evidence_density = 0.0

        # --------------------------------------------------------------
        # Interactions
        # --------------------------------------------------------------

        risk_x_matched = (
            context.risk_probability
            * matched_count
        )

        risk_x_context = (
            context.risk_probability
            * candidate_score
        )

        clinical_x_sdoh = (
            context.clinical_factor_count
            * context.sdoh_factor_count
        )

        evidence_x_matched = (
            context.evidence_count
            * matched_count
        )

        return InterventionRankingFeatures(

            member_id=context.member_id,

            intervention_id=_candidate_id(
                candidate
            ),

            risk_probability=(
                context.risk_probability
            ),

            risk_band_score=(
                context.risk_band_score
            ),

            sdoh_factor_count=(
                context.sdoh_factor_count
            ),

            sdoh_domain_count=(
                context.sdoh_domain_count
            ),

            clinical_factor_count=(
                context.clinical_factor_count
            ),

            evidence_count=(
                context.evidence_count
            ),

            matched_factor_count=(
                matched_count
            ),

            factor_match_ratio=(
                factor_match_ratio
            ),

            intervention_context_score=(
                candidate_score
            ),

            baseline_priority_score=(
                candidate_score
            ),

            evidence_match_count=float(
                len(
                    candidate_evidence
                )
            ),

            evidence_density=(
                evidence_density
            ),

            risk_x_matched_factors=(
                risk_x_matched
            ),

            risk_x_context_score=(
                risk_x_context
            ),

            clinical_x_sdoh=(
                clinical_x_sdoh
            ),

            evidence_x_matched_factors=(
                evidence_x_matched
            ),

            candidate_rank=float(
                candidate_rank
            ),
        )

    # ------------------------------------------------------------------
    # Candidate collection
    # ------------------------------------------------------------------

    def build_many(
        self,
        reasoning_result: Any,
        candidates: Iterable[Any],
    ) -> list[InterventionRankingFeatures]:
        """
        Build feature vectors for all intervention candidates.
        """

        records = []

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):

            records.append(
                self.build(
                    reasoning_result,
                    candidate,
                    candidate_rank=index,
                )
            )

        return records


# ============================================================================
# LAMBDAMART MATRIX
# ============================================================================

FEATURE_NAMES = [
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
]


def to_feature_matrix(
    records: Iterable[
        InterventionRankingFeatures
    ],
) -> list[list[float]]:
    """
    Convert feature records into a numeric matrix.

    The column ordering is fixed by FEATURE_NAMES.
    """

    matrix = []

    for record in records:

        row = [
            float(
                getattr(
                    record,
                    feature_name,
                )
            )
            for feature_name in FEATURE_NAMES
        ]

        matrix.append(
            row
        )

    return matrix


# ============================================================================
# SERIALIZATION
# ============================================================================

def serialize_features(
    records: Iterable[
        InterventionRankingFeatures
    ],
) -> dict[str, Any]:
    """
    Serialize ranking features into a stable JSON structure.
    """

    records = list(
        records
    )

    return {
        "feature_schema_version": (
            FEATURE_SCHEMA_VERSION
        ),

        "feature_names": list(
            FEATURE_NAMES
        ),

        "record_count": len(
            records
        ),

        "records": [
            asdict(
                record
            )
            for record in records
        ],
    }


def save_features(
    records: Iterable[
        InterventionRankingFeatures
    ],
    output_path: str | Path,
) -> Path:
    """
    Save LambdaMART-ready feature data.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = serialize_features(
        records
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


# ============================================================================
# SELF TEST
# ============================================================================

def _self_test() -> None:

    print("=" * 70)

    print(
        "HEALTHLENS LAMBDAMART RANKING FEATURES"
    )

    print("=" * 70)

    # ------------------------------------------------------------------
    # Synthetic reasoning context for structural testing
    # ------------------------------------------------------------------

    reasoning = {
        "member_id": (
            "member:"
            "1e7909f8_39b2_3c7e_1fda_"
            "a6c3256dc061"
        ),

        "risk_probability": (
            0.9999193422541974
        ),

        "risk_band": "Very High",

        "sdoh_factors": [
            "housing_rent_35_plus_pct",
            "snap_households_count_sum",
            "housing_crowded_1_01_to_1_50_pct",
            "straight_no_vehicle_households_beyond_1mi_count_sum",
            "places_routine_checkup_pct",
            "places_obesity_pct",
            "education_less_than_9th_pct",
            "driving_low_income_low_access_tract_count",
            "places_physical_inactivity_pct",
            "places_copd_pct",
            "places_cholesterol_screening_pct",
            "driving_no_vehicle_households_beyond_1mi_count_sum",
            "places_diabetes_pct",
            "housing_crowded_1_51_plus_pct",
            "driving_low_access_population_beyond_1mi_10mi_count_sum",
            "housing_rent_30_to_34_9_pct",
            "places_poor_physical_health_pct",
        ],

        "sdoh_domains": [
            "Housing",
            "Economic Stability",
            "Transportation",
            "Healthcare Access",
            "Neighborhood and Built Environment",
            "Education Access",
        ],

        "clinical_factors": [
            "obesity",
            "diabetes",
            "copd",
            "physical_inactivity",
            "poor_physical_health",
            "cholesterol",
            "routine_checkup",
            "risk_factor_8",
            "risk_factor_9",
        ],

        "evidence_records": [
            "e1",
            "e2",
            "e3",
            "e4",
            "e5",
        ],
    }

    candidates = [

        {
            "intervention_id":
                "INT_HEALTH_ENVIRONMENT",

            "score": 51,

            "matched_factors": [
                "places_obesity_pct",
                "places_physical_inactivity_pct",
                "places_copd_pct",
                "places_diabetes_pct",
                "places_poor_physical_health_pct",
            ],

            "evidence": [
                "e1",
                "e2",
                "e3",
                "e4",
                "e5",
            ],
        },

        {
            "intervention_id":
                "INT_HOUSING_STABILITY",

            "score": 41,

            "matched_factors": [
                "housing_rent_35_plus_pct",
                "housing_crowded_1_01_to_1_50_pct",
                "housing_crowded_1_51_plus_pct",
                "housing_rent_30_to_34_9_pct",
            ],

            "evidence": [
                "e6",
                "e7",
                "e8",
                "e9",
            ],
        },

        {
            "intervention_id":
                "INT_TRANSPORTATION",

            "score": 41,

            "matched_factors": [
                "straight_no_vehicle_households_beyond_1mi_count_sum",
                "driving_low_income_low_access_tract_count",
                "driving_no_vehicle_households_beyond_1mi_count_sum",
                "driving_low_access_population_beyond_1mi_10mi_count_sum",
            ],

            "evidence": [
                "e10",
                "e11",
                "e12",
                "e13",
            ],
        },
    ]

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    context = build_member_context(
        reasoning
    )

    assert (
        context.member_id
    )

    assert (
        context.risk_probability
        > 0.99
    )

    assert (
        context.sdoh_factor_count
        == 17
    )

    print(
        "Member context:              PASS"
    )

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    builder = (
        RankingFeatureBuilder()
    )

    records = builder.build_many(
        reasoning,
        candidates,
    )

    assert len(
        records
    ) == 3

    print(
        "Feature construction:        PASS"
    )

    # ------------------------------------------------------------------
    # Feature validation
    # ------------------------------------------------------------------

    for record in records:

        assert (
            record.member_id
        )

        assert (
            record.intervention_id
        )

        assert (
            0.0
            <= record.risk_probability
            <= 1.0
        )

        assert (
            record.matched_factor_count
            >= 0
        )

        assert (
            record.intervention_context_score
            >= 0
        )

    print(
        "Feature validation:          PASS"
    )

    # ------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------

    matrix = to_feature_matrix(
        records
    )

    assert len(
        matrix
    ) == 3

    assert all(
        len(row)
        == len(FEATURE_NAMES)
        for row in matrix
    )

    print(
        "Feature matrix:              PASS"
    )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    payload = serialize_features(
        records
    )

    assert (
        payload[
            "feature_schema_version"
        ]
        == FEATURE_SCHEMA_VERSION
    )

    assert (
        payload[
            "feature_names"
        ]
        == FEATURE_NAMES
    )

    assert (
        payload[
            "record_count"
        ]
        == 3
    )

    assert all(
        isinstance(
            item,
            dict,
        )
        for item
        in payload[
            "records"
        ]
    )

    print(
        "Serialization:               PASS"
    )

    # ------------------------------------------------------------------
    # Feature ordering
    # ------------------------------------------------------------------

    assert (
        len(FEATURE_NAMES)
        == 17
    )

    print(
        "Feature schema:              PASS"
    )

    print()

    print(
        "Feature columns:"
    )

    for index, name in enumerate(
        FEATURE_NAMES
    ):

        print(
            f"  {index:02d}. {name}"
        )

    print()

    print(
        "LAMBDA MART FEATURE SELF-TEST: PASSED"
    )

    print(
        "=" * 70
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    _self_test()