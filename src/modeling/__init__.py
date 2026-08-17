"""
HealthLens Modeling Package
============================

Public API for member-risk modeling.
"""

from .train_member_risk_model import (
    ModelArtifact,
    MODEL_VERSION,
    TARGET_COLUMN,
    DATASET_FILE,
    MODEL_FILE,
    SCHEMA_FILE,
    METRICS_FILE,
    load_dataset,
    load_feature_schema,
    validate_dataset,
    build_feature_matrix,
    train_pipeline,
    predict_member_risk,
    save_artifacts,
    load_artifacts,
)

__all__ = [
    "ModelArtifact",
    "MODEL_VERSION",
    "TARGET_COLUMN",
    "DATASET_FILE",
    "MODEL_FILE",
    "SCHEMA_FILE",
    "METRICS_FILE",
    "load_dataset",
    "load_feature_schema",
    "validate_dataset",
    "build_feature_matrix",
    "train_pipeline",
    "predict_member_risk",
    "save_artifacts",
    "load_artifacts",
]