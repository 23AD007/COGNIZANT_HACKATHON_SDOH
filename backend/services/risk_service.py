from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


from src.modeling.train_member_risk_model import load_artifacts, predict_member_risk


@lru_cache(maxsize=1)
def _model_artifact():
    """Load the current source model once without retraining it."""
    return load_artifacts()


def get_risk(db: Session, member_id: str) -> dict[str, Any] | None:
    row = db.execute(text("""
        SELECT * FROM member_model_features WHERE member_id = :member_id
    """), {"member_id": member_id}).mappings().first()
    if row is None:
        return None
    return predict_member_risk(dict(row), artifact=_model_artifact())


def get_recommendation(db: Session, member_id: str) -> dict[str, Any] | None:
    row = db.execute(text("""
        SELECT payload_json FROM member_recommendations WHERE member_id = :member_id
    """), {"member_id": member_id}).mappings().first()
    return json.loads(row["payload_json"]) if row is not None else None


def dashboard_summary(db: Session) -> dict[str, Any]:
    members = db.execute(text("SELECT * FROM member_model_features")).mappings()
    risks = [predict_member_risk(dict(member), artifact=_model_artifact()) for member in members]
    risk_bands = Counter(risk["risk_band"] for risk in risks)
    recommendation_count = db.execute(text("SELECT COUNT(*) FROM member_recommendations")).scalar_one()
    return {
        "member_count": len(risks),
        "risk_output_member_count": len(risks),
        "risk_band_counts": dict(sorted(risk_bands.items())),
        "recommendation_member_count": recommendation_count,
    }
