from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
import json

import pandas as pd


# ============================================================================
# HEALTHLENS MEMBER RISK
# ============================================================================
#
# Purpose
# -------
# This module provides the downstream-facing member-risk API.
#
# The trained model itself produces:
#
#     data/processed/member_risk_scores.csv
#
# This module reads that artifact and exposes a stable interface for:
#
#     reasoning
#     prioritization
#     recommendation
#     LambdaMART ranking
#
# IMPORTANT:
# This module does NOT invent SDOH features and does NOT retrain the model.
# The model-training pipeline remains responsible for producing the risk
# scores. This module is the adapter between that artifact and downstream
# layers.
# ============================================================================


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RISK_SCORE_FILE = (
    PROCESSED_DIR / "member_risk_scores.csv"
)

MODEL_DIR = (
    PROCESSED_DIR / "models"
)

MODEL_ARTIFACT = (
    MODEL_DIR / "member_risk_model.pkl"
)

MODEL_SCHEMA = (
    MODEL_DIR / "member_risk_feature_schema.json"
)

MODEL_METRICS = (
    MODEL_DIR / "member_risk_model_metrics.json"
)


# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass(frozen=True)
class MemberRiskScore:
    """
    Canonical member-risk record.

    These fields correspond to the generated member risk score artifact.
    """

    member_id: str

    patient_id: Optional[str]

    county_fips: Optional[float]

    risk_probability: float

    risk_percentile: float

    risk_rank: int

    risk_band: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# MEMBER RISK SCORE ADAPTER
# ============================================================================

class MemberRiskScoreAdapter:
    """
    Adapter around:

        data/processed/member_risk_scores.csv

    Responsibilities
    ----------------
    - load the existing risk-score artifact
    - validate its schema
    - normalize member IDs
    - expose member-level risk records
    - provide downstream reasoning with risk information

    This class deliberately does not retrain the model.
    """

    REQUIRED_COLUMNS = {
        "member_id",
        "risk_probability",
        "risk_percentile",
        "risk_rank",
        "risk_band",
    }

    OPTIONAL_COLUMNS = {
        "patient_id",
        "county_fips",
    }

    DEFAULT_PATH = RISK_SCORE_FILE

    def __init__(
        self,
        risk_score_path: str | Path | None = None,
    ) -> None:

        self.path = Path(
            risk_score_path
            if risk_score_path is not None
            else self.DEFAULT_PATH
        ).resolve()

        self._dataframe = self._load_dataframe()

        self._records = self._build_records()

    # ------------------------------------------------------------------------
    # FILE LOADING
    # ------------------------------------------------------------------------

    def _load_dataframe(self) -> pd.DataFrame:

        if not self.path.exists():

            raise FileNotFoundError(
                "Member risk score file not found:\n"
                f"{self.path}\n\n"
                "Expected artifact:\n"
                f"{RISK_SCORE_FILE}"
            )

        df = pd.read_csv(self.path)

        if df.empty:

            raise ValueError(
                "Member risk score file is empty:\n"
                f"{self.path}"
            )

        missing = (
            self.REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing:

            raise ValueError(
                "Invalid member risk score schema.\n"
                f"Missing columns: {sorted(missing)}\n"
                f"Available columns: {list(df.columns)}"
            )

        return df

    # ------------------------------------------------------------------------
    # MEMBER ID
    # ------------------------------------------------------------------------

    @staticmethod
    def normalize_member_id(
        member_id: Any,
    ) -> str:

        if member_id is None:

            raise ValueError(
                "member_id cannot be None."
            )

        value = str(member_id).strip()

        if not value:

            raise ValueError(
                "member_id cannot be empty."
            )

        return value

    # ------------------------------------------------------------------------
    # OPTIONAL VALUES
    # ------------------------------------------------------------------------

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        value = str(value).strip()

        return value if value else None

    @staticmethod
    def _optional_float(
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        return float(value)

    # ------------------------------------------------------------------------
    # RECORD CREATION
    # ------------------------------------------------------------------------

    def _build_records(
        self,
    ) -> dict[str, MemberRiskScore]:

        records: dict[str, MemberRiskScore] = {}

        for _, row in self._dataframe.iterrows():

            member_id = self.normalize_member_id(
                row["member_id"]
            )

            risk_probability = float(
                row["risk_probability"]
            )

            risk_percentile = float(
                row["risk_percentile"]
            )

            risk_rank = int(
                row["risk_rank"]
            )

            risk_band = str(
                row["risk_band"]
            ).strip()

            # --------------------------------------------------------------
            # VALIDATION
            # --------------------------------------------------------------

            if not (
                0.0
                <= risk_probability
                <= 1.0
            ):

                raise ValueError(
                    "Invalid risk_probability for "
                    f"{member_id}: "
                    f"{risk_probability}"
                )

            if not (
                0.0
                <= risk_percentile
                <= 100.0
            ):

                raise ValueError(
                    "Invalid risk_percentile for "
                    f"{member_id}: "
                    f"{risk_percentile}"
                )

            if risk_rank < 1:

                raise ValueError(
                    "Invalid risk_rank for "
                    f"{member_id}: "
                    f"{risk_rank}"
                )

            if not risk_band:

                raise ValueError(
                    "Empty risk_band for "
                    f"{member_id}"
                )

            patient_id = None

            if "patient_id" in row.index:

                patient_id = (
                    self._optional_string(
                        row["patient_id"]
                    )
                )

            county_fips = None

            if "county_fips" in row.index:

                county_fips = (
                    self._optional_float(
                        row["county_fips"]
                    )
                )

            record = MemberRiskScore(

                member_id=member_id,

                patient_id=patient_id,

                county_fips=county_fips,

                risk_probability=(
                    risk_probability
                ),

                risk_percentile=(
                    risk_percentile
                ),

                risk_rank=risk_rank,

                risk_band=risk_band,
            )

            if member_id in records:

                raise ValueError(
                    "Duplicate member_id detected "
                    f"in risk score file: {member_id}"
                )

            records[member_id] = record

        if not records:

            raise ValueError(
                "No valid member risk records "
                "were loaded."
            )

        return records

    # ------------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------------

    @property
    def dataframe(self) -> pd.DataFrame:

        return self._dataframe.copy()

    @property
    def member_count(self) -> int:

        return len(self._records)

    @property
    def path(self) -> Path:

        return self._path

    @path.setter
    def path(
        self,
        value: Path,
    ) -> None:

        self._path = value

    # ------------------------------------------------------------------------
    # MEMBER LOOKUP
    # ------------------------------------------------------------------------

    def get(
        self,
        member_id: str,
    ) -> Optional[MemberRiskScore]:

        normalized_id = (
            self.normalize_member_id(
                member_id
            )
        )

        return self._records.get(
            normalized_id
        )

    def require(
        self,
        member_id: str,
    ) -> MemberRiskScore:

        record = self.get(member_id)

        if record is None:

            raise KeyError(
                "Member risk score not found: "
                f"{member_id}"
            )

        return record

    def get_member_risk(
        self,
        member_id: str,
    ) -> Optional[dict[str, Any]]:

        record = self.get(member_id)

        if record is None:

            return None

        return record.to_dict()

    # ------------------------------------------------------------------------
    # COLLECTION LOOKUPS
    # ------------------------------------------------------------------------

    def all_scores(
        self,
    ) -> list[MemberRiskScore]:

        return list(
            self._records.values()
        )

    def all_records(
        self,
    ) -> list[dict[str, Any]]:

        return [
            record.to_dict()
            for record in self._records.values()
        ]

    def member_ids(
        self,
    ) -> list[str]:

        return list(
            self._records.keys()
        )

    # ------------------------------------------------------------------------
    # SIMPLE RISK ACCESSORS
    # ------------------------------------------------------------------------

    def get_probability(
        self,
        member_id: str,
    ) -> Optional[float]:

        record = self.get(member_id)

        if record is None:
            return None

        return record.risk_probability

    def get_percentile(
        self,
        member_id: str,
    ) -> Optional[float]:

        record = self.get(member_id)

        if record is None:
            return None

        return record.risk_percentile

    def get_rank(
        self,
        member_id: str,
    ) -> Optional[int]:

        record = self.get(member_id)

        if record is None:
            return None

        return record.risk_rank

    def get_band(
        self,
        member_id: str,
    ) -> Optional[str]:

        record = self.get(member_id)

        if record is None:
            return None

        return record.risk_band

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    def validate(
        self,
    ) -> dict[str, Any]:

        probabilities = [
            record.risk_probability
            for record in self._records.values()
        ]

        percentiles = [
            record.risk_percentile
            for record in self._records.values()
        ]

        ranks = [
            record.risk_rank
            for record in self._records.values()
        ]

        return {

            "valid": True,

            "path": str(
                self.path
            ),

            "member_count": (
                len(self._records)
            ),

            "probability_min": (
                min(probabilities)
            ),

            "probability_max": (
                max(probabilities)
            ),

            "percentile_min": (
                min(percentiles)
            ),

            "percentile_max": (
                max(percentiles)
            ),

            "rank_min": (
                min(ranks)
            ),

            "rank_max": (
                max(ranks)
            ),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_member_risk_scores(
    risk_score_path: str | Path | None = None,
) -> MemberRiskScoreAdapter:

    return MemberRiskScoreAdapter(
        risk_score_path=risk_score_path
    )


def get_member_risk_score(
    member_id: str,
    risk_score_path: str | Path | None = None,
) -> Optional[MemberRiskScore]:

    adapter = MemberRiskScoreAdapter(
        risk_score_path=risk_score_path
    )

    return adapter.get(member_id)


# ============================================================================
# COMPATIBILITY HELPERS
# ============================================================================

def get_member_risk(
    member_id: str,
    risk_score_path: str | Path | None = None,
) -> Optional[dict[str, Any]]:

    adapter = MemberRiskScoreAdapter(
        risk_score_path=risk_score_path
    )

    return adapter.get_member_risk(
        member_id
    )


def get_risk_probability(
    member_id: str,
    risk_score_path: str | Path | None = None,
) -> Optional[float]:

    adapter = MemberRiskScoreAdapter(
        risk_score_path=risk_score_path
    )

    return adapter.get_probability(
        member_id
    )


# ============================================================================
# SELF TEST
# ============================================================================

def _self_test() -> None:

    print("=" * 70)
    print("HEALTHLENS MEMBER RISK SCORE ADAPTER")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------------

    adapter = MemberRiskScoreAdapter()

    print(
        "Risk score file loading:      PASS"
    )

    # ------------------------------------------------------------------------
    # MEMBER COUNT
    # ------------------------------------------------------------------------

    print(
        f"Members detected:             "
        f"{adapter.member_count}"
    )

    assert adapter.member_count > 0

    # ------------------------------------------------------------------------
    # SCHEMA
    # ------------------------------------------------------------------------

    validation = (
        adapter.validate()
    )

    assert validation["valid"] is True

    print(
        "Risk score schema:            PASS"
    )

    # ------------------------------------------------------------------------
    # ALL MEMBERS
    # ------------------------------------------------------------------------

    records = (
        adapter.all_scores()
    )

    assert len(records) == (
        adapter.member_count
    )

    print(
        "All member retrieval:         PASS"
    )

    # ------------------------------------------------------------------------
    # TEST MEMBER
    # ------------------------------------------------------------------------

    test_member_id = (
        "1e7909f8-39b2-3c7e-1fda-a6c3256dc061"
    )

    record = adapter.get(
        test_member_id
    )

    # The adapter must work regardless of whether
    # this specific member is present in a future
    # artifact.

    if record is not None:

        print(
            "Member risk lookup:           PASS"
        )

        # --------------------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------------------

        assert (
            0.0
            <= record.risk_probability
            <= 1.0
        )

        print(
            "Risk probability validation:  PASS"
        )

        # --------------------------------------------------------------
        # RANK
        # --------------------------------------------------------------

        assert (
            record.risk_rank >= 1
        )

        print(
            "Risk rank validation:         PASS"
        )

        # --------------------------------------------------------------
        # PERCENTILE
        # --------------------------------------------------------------

        assert (
            0.0
            <= record.risk_percentile
            <= 100.0
        )

        print(
            "Risk percentile validation:   PASS"
        )

        # --------------------------------------------------------------
        # BAND
        # --------------------------------------------------------------

        assert record.risk_band

        print(
            "Risk band validation:         PASS"
        )

        # --------------------------------------------------------------
        # SERIALIZATION
        # --------------------------------------------------------------

        serialized = (
            record.to_dict()
        )

        assert (
            serialized["member_id"]
            == test_member_id
        )

        json.dumps(
            serialized
        )

        print(
            "Risk serialization:           PASS"
        )

        # --------------------------------------------------------------
        # OUTPUT
        # --------------------------------------------------------------

        print()
        print("TEST MEMBER")
        print("-" * 70)

        print(
            f"Member ID:          "
            f"{record.member_id}"
        )

        print(
            f"Risk probability:   "
            f"{record.risk_probability:.6f}"
        )

        print(
            f"Risk percentile:    "
            f"{record.risk_percentile:.2f}"
        )

        print(
            f"Risk rank:          "
            f"{record.risk_rank}"
        )

        print(
            f"Risk band:          "
            f"{record.risk_band}"
        )

    else:

        print(
            "Member risk lookup:           "
            "PASS (test member absent)"
        )

    # ------------------------------------------------------------------------
    # MISSING MEMBER
    # ------------------------------------------------------------------------

    missing = adapter.get(
        "__NON_EXISTING_MEMBER__"
    )

    assert missing is None

    print(
        "Missing-member handling:      PASS"
    )

    # ------------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------------

    print()
    print(
        "RISK SCORE ADAPTER SELF-TEST: PASSED"
    )
    print("=" * 70)


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    _self_test()