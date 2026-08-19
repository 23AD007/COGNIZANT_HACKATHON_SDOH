# ============================================================================
# HEALTHLENS — LAMBDAMART RANKER
# ============================================================================
#
# Purpose:
#   Train and execute a LambdaMART learning-to-rank model for intervention
#   ranking.
#
# Pipeline:
#
#   Contextual Reasoner
#          ↓
#   Intervention Prioritizer
#          ↓
#   LambdaMART Feature Builder
#          ↓
#   17 ranking features
#          ↓
#   LambdaMART
#          ↓
#   Ranked interventions
#          ↓
#   Recommendation Engine
#
# IMPORTANT:
#   The current project can use weak/synthetic relevance labels generated
#   from the existing prioritization score. These labels are intended for
#   pipeline development only. Replace them with outcome-based labels when
#   real intervention outcome data becomes available.
#
# ============================================================================

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "models"
)

DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "lambdamart_ranker.pkl"

DEFAULT_FEATURE_COLUMNS = [
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


# ============================================================================
# OPTIONAL LIGHTGBM
# ============================================================================

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True

except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(frozen=True)
class RankingCandidate:
    """
    One intervention candidate and its LambdaMART feature vector.
    """

    intervention_id: str
    intervention_name: str
    domain: str
    features: dict[str, float]
    baseline_score: float = 0.0
    baseline_rank: int = 0
    matched_factor_count: int = 0
    evidence_match_count: int = 0


@dataclass(frozen=True)
class RankedCandidate:
    """
    LambdaMART ranking result.
    """

    intervention_id: str
    intervention_name: str
    domain: str
    model_score: float
    rank: int
    baseline_score: float
    baseline_rank: int
    matched_factor_count: int
    evidence_match_count: int


@dataclass
class LambdaMARTTrainingExample:
    """
    One training row.

    group_id:
        Identifies one member/query group.

    relevance:
        Ranking relevance label.

    features:
        Exactly the 17 LambdaMART features.
    """

    group_id: str
    intervention_id: str
    relevance: int
    features: dict[str, float]


@dataclass
class LambdaMARTModelMetadata:
    """
    Serializable metadata associated with a trained model.
    """

    model_version: str
    schema_version: str
    feature_columns: list[str]
    objective: str
    metric: str
    num_features: int
    training_examples: int
    training_groups: int
    backend: str


@dataclass
class LambdaMARTRankingResult:
    """
    Complete ranking response.
    """

    member_id: str | None
    model_version: str
    feature_columns: list[str]
    candidates: list[RankedCandidate]
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "model_version": self.model_version,
            "feature_columns": list(self.feature_columns),
            "backend": self.backend,
            "candidates": [
                asdict(candidate)
                for candidate in self.candidates
            ],
        }


# ============================================================================
# FEATURE VALIDATION
# ============================================================================


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to a finite float.
    """

    try:
        number = float(value)

    except (TypeError, ValueError):
        return default

    if not math.isfinite(number):
        return default

    return number


def validate_feature_columns(
    feature_columns: Sequence[str],
) -> None:
    """
    Validate the exact LambdaMART feature schema.
    """

    actual = list(feature_columns)
    expected = list(DEFAULT_FEATURE_COLUMNS)

    if actual != expected:
        missing = [
            feature
            for feature in expected
            if feature not in actual
        ]

        extra = [
            feature
            for feature in actual
            if feature not in expected
        ]

        raise ValueError(
            "LambdaMART feature schema mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}\n"
            f"Missing:  {missing}\n"
            f"Extra:    {extra}"
        )


def normalize_features(
    features: Mapping[str, Any],
) -> dict[str, float]:
    """
    Convert an arbitrary feature mapping into the canonical 17-feature schema.

    Missing values are set to zero.
    Extra features are ignored.
    """

    normalized: dict[str, float] = {}

    for feature in DEFAULT_FEATURE_COLUMNS:
        normalized[feature] = _safe_float(
            features.get(feature, 0.0)
        )

    return normalized


def feature_vector(
    features: Mapping[str, Any],
) -> list[float]:
    """
    Return the ordered feature vector expected by LambdaMART.
    """

    normalized = normalize_features(features)

    return [
        normalized[feature]
        for feature in DEFAULT_FEATURE_COLUMNS
    ]


# ============================================================================
# RELEVANCE LABEL GENERATION
# ============================================================================


def priority_score_to_relevance(
    priority_score: float,
) -> int:
    """
    Convert the existing intervention priority score into a weak relevance
    label for LambdaMART development.

    This is NOT an outcome label.

    Score:
        >= 80  -> 4 Critical
        >= 60  -> 3 High
        >= 30  -> 2 Moderate
        >  0   -> 1 Low
        <= 0   -> 0 Not relevant
    """

    score = _safe_float(priority_score)

    if score >= 80:
        return 4

    if score >= 60:
        return 3

    if score >= 30:
        return 2

    if score > 0:
        return 1

    return 0


def build_training_example(
    group_id: str,
    candidate: RankingCandidate,
) -> LambdaMARTTrainingExample:
    """
    Convert an existing intervention candidate into a training example.
    """

    return LambdaMARTTrainingExample(
        group_id=str(group_id),
        intervention_id=candidate.intervention_id,
        relevance=priority_score_to_relevance(
            candidate.baseline_score
        ),
        features=normalize_features(candidate.features),
    )


def build_training_examples(
    grouped_candidates: Mapping[
        str,
        Sequence[RankingCandidate],
    ],
) -> list[LambdaMARTTrainingExample]:
    """
    Build weakly supervised LambdaMART examples.

    grouped_candidates:
        {
            member_id: [
                candidate1,
                candidate2,
                ...
            ]
        }
    """

    examples: list[LambdaMARTTrainingExample] = []

    for group_id, candidates in grouped_candidates.items():

        for candidate in candidates:

            examples.append(
                build_training_example(
                    group_id=group_id,
                    candidate=candidate,
                )
            )

    return examples


# ============================================================================
# LAMBDAMART RANKER
# ============================================================================


class LambdaMARTRanker:
    """
    HealthLens LambdaMART ranking engine.

    LightGBM is used when available and a trained model is loaded.

    Before a model is trained, a deterministic baseline ranking is used so
    that the complete pipeline remains executable.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        feature_columns: Sequence[str] | None = None,
    ) -> None:

        self.feature_columns = list(
            feature_columns
            if feature_columns is not None
            else DEFAULT_FEATURE_COLUMNS
        )

        validate_feature_columns(
            self.feature_columns
        )

        self.model_path = Path(
            model_path
            if model_path is not None
            else DEFAULT_MODEL_PATH
        )

        self.model: Any = None
        self.metadata: LambdaMARTModelMetadata | None = None

        if self.model_path.exists():
            self.load()

    # ------------------------------------------------------------------------
    # BACKEND
    # ------------------------------------------------------------------------

    @property
    def backend(self) -> str:
        if self.model is not None:
            return "lightgbm_lambdamart"

        return "deterministic_baseline"

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------

    def train(
        self,
        examples: Sequence[LambdaMARTTrainingExample],
        num_boost_round: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_data_in_leaf: int = 2,
    ) -> LambdaMARTModelMetadata:
        """
        Train a LightGBM LambdaMART model.

        The training data must contain multiple candidates per group.

        Each group represents one member/query.
        """

        if not examples:
            raise ValueError(
                "Cannot train LambdaMART with zero examples."
            )

        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError(
                "LightGBM is not installed. "
                "Install it with: pip install lightgbm"
            )

        rows: list[list[float]] = []
        labels: list[int] = []
        groups: list[int] = []

        grouped: dict[str, list[LambdaMARTTrainingExample]] = {}

        for example in examples:

            normalized = normalize_features(
                example.features
            )

            rows.append(
                [
                    normalized[feature]
                    for feature in self.feature_columns
                ]
            )

            labels.append(
                int(example.relevance)
            )

            grouped.setdefault(
                str(example.group_id),
                [],
            ).append(example)

        for group_examples in grouped.values():

            if len(group_examples) < 2:
                raise ValueError(
                    "LambdaMART requires at least two candidates "
                    "per ranking group. "
                    f"Group contains {len(group_examples)} candidate."
                )

            groups.append(
                len(group_examples)
            )

        dataset = lgb.Dataset(
            rows,
            label=labels,
            feature_name=self.feature_columns,
            group=groups,
            free_raw_data=False,
        )

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_at": [1, 3, 5],
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "verbosity": -1,
            "seed": 42,
            "feature_pre_filter": False,
        }

        self.model = lgb.train(
            params,
            dataset,
            num_boost_round=num_boost_round,
        )

        self.metadata = LambdaMARTModelMetadata(
            model_version=MODEL_VERSION,
            schema_version=SCHEMA_VERSION,
            feature_columns=list(
                self.feature_columns
            ),
            objective="lambdarank",
            metric="ndcg",
            num_features=len(
                self.feature_columns
            ),
            training_examples=len(examples),
            training_groups=len(grouped),
            backend="lightgbm_lambdamart",
        )

        self.save()

        return self.metadata

    # ------------------------------------------------------------------------
    # BASELINE SCORE
    # ------------------------------------------------------------------------

    @staticmethod
    def _baseline_ranking_score(
        candidate: RankingCandidate,
    ) -> float:
        """
        Deterministic fallback score.

        This mirrors the information already present in the feature vector
        without pretending that an untrained LambdaMART model exists.
        """

        features = normalize_features(
            candidate.features
        )

        score = (
            0.40
            * features["baseline_priority_score"]
        )

        score += (
            0.20
            * features["intervention_context_score"]
        )

        score += (
            0.15
            * features["matched_factor_count"]
        )

        score += (
            0.10
            * features["evidence_match_count"]
        )

        score += (
            0.10
            * features["risk_x_matched_factors"]
        )

        score += (
            0.05
            * features["risk_x_context_score"]
        )

        return float(score)

    # ------------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------------

    def predict_scores(
        self,
        candidates: Sequence[RankingCandidate],
    ) -> list[float]:
        """
        Generate ranking scores for candidates.
        """

        if not candidates:
            return []

        matrix = [
            feature_vector(
                candidate.features
            )
            for candidate in candidates
        ]

        if self.model is not None:

            predictions = self.model.predict(
                matrix
            )

            return [
                _safe_float(
                    prediction
                )
                for prediction in predictions
            ]

        return [
            self._baseline_ranking_score(
                candidate
            )
            for candidate in candidates
        ]

    # ------------------------------------------------------------------------
    # RANK
    # ------------------------------------------------------------------------

    def rank(
        self,
        candidates: Sequence[RankingCandidate],
        member_id: str | None = None,
    ) -> LambdaMARTRankingResult:
        """
        Rank intervention candidates.
        """

        if not candidates:
            return LambdaMARTRankingResult(
                member_id=member_id,
                model_version=MODEL_VERSION,
                feature_columns=list(
                    self.feature_columns
                ),
                candidates=[],
                backend=self.backend,
            )

        scores = self.predict_scores(
            candidates
        )

        ranked_pairs = list(
            zip(
                candidates,
                scores,
            )
        )

        ranked_pairs.sort(
            key=lambda pair: (
                -pair[1],
                -pair[0].baseline_score,
                pair[0].baseline_rank
                if pair[0].baseline_rank > 0
                else 999999,
                pair[0].intervention_id,
            )
        )

        ranked: list[RankedCandidate] = []

        for rank_number, (
            candidate,
            score,
        ) in enumerate(
            ranked_pairs,
            start=1,
        ):

            ranked.append(
                RankedCandidate(
                    intervention_id=(
                        candidate.intervention_id
                    ),
                    intervention_name=(
                        candidate.intervention_name
                    ),
                    domain=candidate.domain,
                    model_score=float(score),
                    rank=rank_number,
                    baseline_score=float(
                        candidate.baseline_score
                    ),
                    baseline_rank=int(
                        candidate.baseline_rank
                    ),
                    matched_factor_count=int(
                        candidate.matched_factor_count
                    ),
                    evidence_match_count=int(
                        candidate.evidence_match_count
                    ),
                )
            )

        return LambdaMARTRankingResult(
            member_id=member_id,
            model_version=MODEL_VERSION,
            feature_columns=list(
                self.feature_columns
            ),
            candidates=ranked,
            backend=self.backend,
        )

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    def save(
        self,
        path: str | Path | None = None,
    ) -> Path:
        """
        Save the trained model and metadata.
        """

        output_path = Path(
            path
            if path is not None
            else self.model_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model_version": MODEL_VERSION,
            "schema_version": SCHEMA_VERSION,
            "feature_columns": list(
                self.feature_columns
            ),
            "metadata": (
                asdict(self.metadata)
                if self.metadata is not None
                else None
            ),
            "model": self.model,
        }

        with output_path.open(
            "wb"
        ) as handle:

            pickle.dump(
                payload,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        self.model_path = output_path

        return output_path

    # ------------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------------

    def load(
        self,
        path: str | Path | None = None,
    ) -> None:
        """
        Load a previously trained model.
        """

        input_path = Path(
            path
            if path is not None
            else self.model_path
        )

        if not input_path.exists():
            raise FileNotFoundError(
                f"LambdaMART model not found: "
                f"{input_path}"
            )

        with input_path.open(
            "rb"
        ) as handle:

            payload = pickle.load(
                handle
            )

        loaded_features = payload.get(
            "feature_columns",
            [],
        )

        validate_feature_columns(
            loaded_features
        )

        if list(loaded_features) != list(
            self.feature_columns
        ):
            raise ValueError(
                "Loaded LambdaMART model feature "
                "schema does not match the current schema."
            )

        self.model = payload.get(
            "model"
        )

        raw_metadata = payload.get(
            "metadata"
        )

        if raw_metadata:

            self.metadata = (
                LambdaMARTModelMetadata(
                    **raw_metadata
                )
            )

        self.model_path = input_path


# ============================================================================
# JSON SERIALIZATION
# ============================================================================


def serialize_ranking_result(
    result: LambdaMARTRankingResult,
) -> dict[str, Any]:
    """
    Serialize ranking result to JSON-compatible dictionary.
    """

    return result.to_dict()


def save_ranking_result(
    result: LambdaMARTRankingResult,
    path: str | Path,
) -> Path:
    """
    Save ranking result.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            serialize_ranking_result(
                result
            ),
            handle,
            indent=2,
        )

    return output_path


# ============================================================================
# DEMO CANDIDATES
# ============================================================================


def _build_test_candidates() -> list[RankingCandidate]:
    """
    Build realistic test candidates using the same 17-feature schema already
    validated by the HealthLens LambdaMART feature stage.
    """

    raw_candidates = [
        {
            "intervention_id": "INT_HEALTH_ENVIRONMENT",
            "intervention_name": (
                "Community health and healthy-environment support"
            ),
            "domain": "Neighborhood and Built Environment",
            "baseline_score": 51.0,
            "baseline_rank": 1,
            "matched_factor_count": 5,
            "evidence_match_count": 5,
            "context_score": 51.0,
        },
        {
            "intervention_id": "INT_HOUSING_STABILITY",
            "intervention_name": (
                "Housing stability and housing-support referral"
            ),
            "domain": "Housing",
            "baseline_score": 41.0,
            "baseline_rank": 2,
            "matched_factor_count": 4,
            "evidence_match_count": 4,
            "context_score": 41.0,
        },
        {
            "intervention_id": "INT_TRANSPORTATION",
            "intervention_name": (
                "Transportation assistance for healthcare access"
            ),
            "domain": "Transportation",
            "baseline_score": 41.0,
            "baseline_rank": 3,
            "matched_factor_count": 4,
            "evidence_match_count": 4,
            "context_score": 41.0,
        },
        {
            "intervention_id": "INT_HEALTHCARE_ACCESS",
            "intervention_name": (
                "Healthcare navigation and preventive-care support"
            ),
            "domain": "Healthcare Access",
            "baseline_score": 21.0,
            "baseline_rank": 4,
            "matched_factor_count": 2,
            "evidence_match_count": 2,
            "context_score": 21.0,
        },
        {
            "intervention_id": "INT_ECONOMIC_BENEFITS",
            "intervention_name": (
                "Financial assistance and benefits navigation"
            ),
            "domain": "Economic Stability",
            "baseline_score": 11.0,
            "baseline_rank": 5,
            "matched_factor_count": 1,
            "evidence_match_count": 1,
            "context_score": 11.0,
        },
        {
            "intervention_id": "INT_EDUCATION_SUPPORT",
            "intervention_name": (
                "Health education and literacy support"
            ),
            "domain": "Education Access",
            "baseline_score": 11.0,
            "baseline_rank": 6,
            "matched_factor_count": 1,
            "evidence_match_count": 1,
            "context_score": 11.0,
        },
    ]

    candidates: list[RankingCandidate] = []

    risk_probability = 0.9999193422541974
    risk_band_score = 4.0

    sdoh_factor_count = 17.0
    sdoh_domain_count = 6.0
    clinical_factor_count = 9.0
    evidence_count = 17.0

    for item in raw_candidates:

        matched = float(
            item["matched_factor_count"]
        )

        evidence = float(
            item["evidence_match_count"]
        )

        context_score = float(
            item["context_score"]
        )

        features = {
            "risk_probability": risk_probability,
            "risk_band_score": risk_band_score,
            "sdoh_factor_count": sdoh_factor_count,
            "sdoh_domain_count": sdoh_domain_count,
            "clinical_factor_count": clinical_factor_count,
            "evidence_count": evidence_count,
            "matched_factor_count": matched,
            "factor_match_ratio": (
                matched / sdoh_factor_count
            ),
            "intervention_context_score": context_score,
            "baseline_priority_score": float(
                item["baseline_score"]
            ),
            "evidence_match_count": evidence,
            "evidence_density": (
                evidence / max(1.0, matched)
            ),
            "risk_x_matched_factors": (
                risk_probability * matched
            ),
            "risk_x_context_score": (
                risk_probability * context_score
            ),
            "clinical_x_sdoh": (
                clinical_factor_count
                * sdoh_factor_count
            ),
            "evidence_x_matched_factors": (
                evidence * matched
            ),
            "candidate_rank": float(
                item["baseline_rank"]
            ),
        }

        candidates.append(
            RankingCandidate(
                intervention_id=str(
                    item["intervention_id"]
                ),
                intervention_name=str(
                    item["intervention_name"]
                ),
                domain=str(
                    item["domain"]
                ),
                features=features,
                baseline_score=float(
                    item["baseline_score"]
                ),
                baseline_rank=int(
                    item["baseline_rank"]
                ),
                matched_factor_count=int(
                    item["matched_factor_count"]
                ),
                evidence_match_count=int(
                    item["evidence_match_count"]
                ),
            )
        )

    return candidates


# ============================================================================
# SELF TEST
# ============================================================================


def _self_test() -> None:

    print("=" * 70)
    print(
        "HEALTHLENS LAMBDAMART RANKING ENGINE"
    )
    print("=" * 70)

    # ------------------------------------------------------------------------
    # FEATURE SCHEMA
    # ------------------------------------------------------------------------

    validate_feature_columns(
        DEFAULT_FEATURE_COLUMNS
    )

    print(
        "Feature schema:                 PASS"
    )

    # ------------------------------------------------------------------------
    # CANDIDATES
    # ------------------------------------------------------------------------

    candidates = _build_test_candidates()

    assert len(candidates) == 6

    print(
        "Candidate construction:         PASS"
    )

    # ------------------------------------------------------------------------
    # FEATURE VALIDATION
    # ------------------------------------------------------------------------

    for candidate in candidates:

        assert (
            list(candidate.features.keys())
            == DEFAULT_FEATURE_COLUMNS
        )

        vector = feature_vector(
            candidate.features
        )

        assert len(vector) == 17

        assert all(
            isinstance(value, float)
            for value in vector
        )

    print(
        "Feature validation:             PASS"
    )

    # ------------------------------------------------------------------------
    # TRAINING LABELS
    # ------------------------------------------------------------------------

    grouped_candidates = {
        "test-member": candidates
    }

    examples = build_training_examples(
        grouped_candidates
    )

    assert len(examples) == 6

    assert all(
        0 <= example.relevance <= 4
        for example in examples
    )

    print(
        "Training label construction:    PASS"
    )

    # ------------------------------------------------------------------------
    # RANKER
    # ------------------------------------------------------------------------

    ranker = LambdaMARTRanker(
        model_path=(
            Path("data")
            / "processed"
            / "models"
            / "healthlens_test_lambdamart.pkl"
        )
    )

    print(
        "Ranker construction:            PASS"
    )

    # ------------------------------------------------------------------------
    # RANKING
    # ------------------------------------------------------------------------

    result = ranker.rank(
        candidates,
        member_id=(
            "member:1e7909f8_39b2_3c7e_1fda_a6c3256dc061"
        ),
    )

    assert len(
        result.candidates
    ) == len(candidates)

    assert [
        candidate.rank
        for candidate in result.candidates
    ] == list(
        range(
            1,
            len(candidates) + 1,
        )
    )

    print(
        "Ranking execution:              PASS"
    )

    # ------------------------------------------------------------------------
    # SCORE VALIDATION
    # ------------------------------------------------------------------------

    assert all(
        math.isfinite(
            candidate.model_score
        )
        for candidate in result.candidates
    )

    print(
        "Score validation:               PASS"
    )

    # ------------------------------------------------------------------------
    # UNIQUE RANKS
    # ------------------------------------------------------------------------

    ranks = [
        candidate.rank
        for candidate in result.candidates
    ]

    assert len(ranks) == len(
        set(ranks)
    )

    print(
        "Rank uniqueness:                PASS"
    )

    # ------------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------------

    serialized = result.to_dict()

    assert isinstance(
        serialized,
        dict,
    )

    assert (
        serialized["model_version"]
        == MODEL_VERSION
    )

    assert (
        serialized["feature_columns"]
        == DEFAULT_FEATURE_COLUMNS
    )

    assert len(
        serialized["candidates"]
    ) == 6

    print(
        "Serialization:                  PASS"
    )

    # ------------------------------------------------------------------------
    # BACKEND
    # ------------------------------------------------------------------------

    assert result.backend in {
        "lightgbm_lambdamart",
        "deterministic_baseline",
    }

    print(
        "Backend validation:             PASS"
    )

    # ------------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------------

    output_path = (
        Path("data")
        / "processed"
        / "ranking"
        / "healthlens_lambdamart_test_result.json"
    )

    save_ranking_result(
        result,
        output_path,
    )

    assert output_path.exists()

    print(
        "Output serialization:           PASS"
    )

    # ------------------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------------------

    print()
    print(
        "LAMBDA MART RANKING RESULT"
    )
    print("=" * 70)

    for candidate in result.candidates:

        print(
            f"{candidate.rank}. "
            f"{candidate.intervention_id} | "
            f"{candidate.intervention_name} | "
            f"score: {candidate.model_score:.4f}"
        )

    print()
    print(
        f"Backend: {result.backend}"
    )

    print()
    print(
        "LAMBDA MART RANKING SELF-TEST: PASSED"
    )

    print("=" * 70)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "MODEL_VERSION",
    "SCHEMA_VERSION",
    "DEFAULT_FEATURE_COLUMNS",
    "RankingCandidate",
    "RankedCandidate",
    "LambdaMARTTrainingExample",
    "LambdaMARTModelMetadata",
    "LambdaMARTRankingResult",
    "LambdaMARTRanker",
    "validate_feature_columns",
    "normalize_features",
    "feature_vector",
    "priority_score_to_relevance",
    "build_training_example",
    "build_training_examples",
    "serialize_ranking_result",
    "save_ranking_result",
]


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    _self_test()