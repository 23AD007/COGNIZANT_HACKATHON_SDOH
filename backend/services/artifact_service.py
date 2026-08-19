"""In-memory API views backed directly by the processed pipeline artifacts."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.knowledge_base.registry import build_registry
from src.modeling.train_member_risk_model import load_artifacts, predict_member_risk
from src.prioritization.lambdamart_recommendation_integration import (
    _load_lambda_components,
    build_lambda_features,
    extract_prioritized_candidates,
    load_healthlens_graph,
    normalize_ranked_result,
    resolve_graph_member,
    run_lambda_ranker,
    run_prioritization,
    run_reasoning,
    run_recommendation_engine,
)
from src.reasoning.contextual_reasoner import ContextualReasoner
from src.reasoning.member_risk_integration import inject_risk_into_member_node
from src.recommendations.recommendation_engine import serialize_recommendations


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MEMBER_FEATURES = PROCESSED_DIR / "member_model_features.csv"
SDOH_FEATURES = PROCESSED_DIR / "member_sdoh_features.csv"


@lru_cache(maxsize=1)
def _members() -> pd.DataFrame:
    return pd.read_csv(MEMBER_FEATURES, dtype={"member_id": str})


@lru_cache(maxsize=1)
def _sdoh() -> pd.DataFrame:
    return pd.read_csv(SDOH_FEATURES, dtype={"member_id": str})


@lru_cache(maxsize=1)
def _model() -> Any:
    return load_artifacts()


@lru_cache(maxsize=1)
def _graph() -> dict[str, Any]:
    return load_healthlens_graph()


@lru_cache(maxsize=1)
def _registry() -> Any:
    return build_registry()


def _clean(record: dict[str, Any]) -> dict[str, Any]:
    return {key: (None if pd.isna(value) else value) for key, value in record.items()}


def _member_row(member_id: str) -> dict[str, Any] | None:
    matches = _members().loc[_members()["member_id"] == str(member_id)]
    return _clean(matches.iloc[0].to_dict()) if not matches.empty else None


def list_members() -> list[dict[str, Any]]:
    fields = ["member_id", "age", "gender", "race", "ethnicity", "city", "state"]
    return [_clean(row) for row in _members().loc[:, fields].sort_values("member_id").to_dict("records")]


def get_member(member_id: str) -> dict[str, Any] | None:
    row = _member_row(member_id)
    if row is None:
        return None
    return {key: row.get(key) for key in ("member_id", "age", "gender", "race", "ethnicity", "city", "state")}


def get_sdoh(member_id: str) -> dict[str, Any] | None:
    matches = _sdoh().loc[_sdoh()["member_id"] == str(member_id)]
    return _clean(matches.iloc[0].to_dict()) if not matches.empty else None


def get_clinical(member_id: str) -> dict[str, Any] | None:
    row = _member_row(member_id)
    if row is None:
        return None
    fields = [key for key in row if key == "member_id" or key == "target_inpatient_any" or key.startswith("clinical_")]
    return {key: row[key] for key in fields}


def get_location(member_id: str) -> dict[str, Any] | None:
    row = _member_row(member_id)
    if row is None:
        return None
    location = {key: row.get(key) for key in ("member_id", "lat", "lon", "county", "county_fips", "city", "state")}
    if location["county_fips"] is not None:
        location["county_fips"] = str(location["county_fips"]).zfill(5)
    return location


def get_risk(member_id: str) -> dict[str, Any] | None:
    row = _member_row(member_id)
    return predict_member_risk(row, artifact=_model()) if row is not None else None


def get_recommendation(member_id: str) -> dict[str, Any] | None:
    """Run the persisted model and recommendation pipeline for one real member."""
    row = _member_row(member_id)
    if row is None:
        return None

    risk = predict_member_risk(row, artifact=_model())
    graph_member_id = resolve_graph_member(_graph(), str(member_id))
    graph = inject_risk_into_member_node(deepcopy(_graph()), graph_member_id, risk)
    reasoning = run_reasoning(ContextualReasoner(graph=graph, registry=_registry()), graph_member_id)
    prioritization = run_prioritization(reasoning)
    candidates = extract_prioritized_candidates(prioritization, reasoning)
    feature_builder, ranker = _load_lambda_components()
    features = build_lambda_features(feature_builder, candidates, reasoning, risk["risk_probability"], risk["risk_band"])
    ranked = normalize_ranked_result(run_lambda_ranker(ranker, features, candidates), candidates)
    result = serialize_recommendations(run_recommendation_engine(ranked, reasoning, prioritization))
    result["member_id"] = str(member_id)
    return result


def dashboard_summary() -> dict[str, Any]:
    risks = [predict_member_risk(row, artifact=_model()) for row in _members().to_dict("records")]
    return {"member_count": len(risks), "risk_output_member_count": len(risks), "risk_band_counts": dict(sorted(Counter(item["risk_band"] for item in risks).items())), "recommendation_member_count": None}
