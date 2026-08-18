"""Inference-only facade for the existing trained member-risk model."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .model_artifacts import ModelArtifact, load_member_risk_artifact
from .train_member_risk_model import predict_member_risk as _predict_member_risk


def predict_member_risk(member_row: pd.Series | dict[str, Any], artifact: ModelArtifact | None = None) -> dict[str, Any]:
    """Predict one member using the existing training artifact schema."""
    return _predict_member_risk(member_row, artifact or load_member_risk_artifact())


__all__ = ["predict_member_risk"]
