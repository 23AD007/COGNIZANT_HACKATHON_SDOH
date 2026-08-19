"""Load the current pipeline artifacts into the local development database.

Production deployments are expected to provision equivalent tables themselves.
The SQLite path is deliberately a convenience view over the processed artifacts,
not a second source of truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# The member-model file is the authoritative member-level source selected by
# src.modeling.train_member_risk_model._resolve_dataset_path().
ARTIFACT_TABLES = {
    "member_model_features": "member_model_features.csv",
    "member_sdoh_features": "member_sdoh_features.csv",
}

RECOMMENDATION_ARTIFACT = (
    PROCESSED_DIR / "recommendations" / "healthlens_lambdamart_recommendations.json"
)


def initialize_development_database(engine: Engine) -> None:
    """Refresh SQLite tables from the current processed pipeline artifacts."""
    if engine.dialect.name != "sqlite":
        return

    frames: dict[str, pd.DataFrame] = {}
    for table_name, filename in ARTIFACT_TABLES.items():
        path = PROCESSED_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Required pipeline artifact is missing: {path}")
        frame = pd.read_csv(path)
        if "member_id" not in frame.columns:
            raise ValueError(f"{path.name} has no member_id column")
        if not frame["member_id"].is_unique:
            raise ValueError(f"{path.name} contains duplicate member_id values")
        frames[table_name] = frame

    model_ids = set(frames["member_model_features"]["member_id"])
    sdoh_ids = set(frames["member_sdoh_features"]["member_id"])
    if model_ids != sdoh_ids:
        raise ValueError(
            "Member-model and SDOH artifacts have mismatched member_id coverage: "
            f"model_only={len(model_ids - sdoh_ids)}, sdoh_only={len(sdoh_ids - model_ids)}"
        )

    model_frame = frames["member_model_features"]
    clinical_columns = [
        column for column in model_frame.columns
        if column == "member_id" or column == "target_inpatient_any"
        or column.startswith("clinical_")
    ]
    frames["member_clinical_features"] = model_frame[clinical_columns].copy()

    if not RECOMMENDATION_ARTIFACT.exists():
        raise FileNotFoundError(
            f"Current LambdaMART recommendation artifact is missing: {RECOMMENDATION_ARTIFACT}"
        )
    recommendation = json.loads(RECOMMENDATION_ARTIFACT.read_text(encoding="utf-8"))
    recommendation_member_id = str(recommendation.get("member_id", "")).strip()
    if not recommendation_member_id:
        raise ValueError("LambdaMART recommendation artifact has no member_id")
    frames["member_recommendations"] = pd.DataFrame([{
        "member_id": recommendation_member_id,
        "payload_json": json.dumps(recommendation),
    }])

    with engine.begin() as connection:
        for table_name, frame in frames.items():
            frame.to_sql(table_name, connection, if_exists="replace", index=False)
            connection.execute(
                text(f'CREATE INDEX IF NOT EXISTS "ix_{table_name}_member_id" '
                     f'ON "{table_name}" (member_id)')
