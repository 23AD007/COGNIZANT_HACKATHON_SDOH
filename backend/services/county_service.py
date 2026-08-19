"""County views calculated from the current member runtime artifacts.

``county_risk_scores.csv`` is deliberately not read here. County risk is a
population aggregation of aligned current member-model and member-risk files.
County interventions are population priorities, not member reasoner output.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
COUNTY_ARTIFACTS = {
    "county_features": PROCESSED_DIR / "county_features.csv",
    "sdoh_features": PROCESSED_DIR / "county_sdoh_features.csv",
    "sram_features": PROCESSED_DIR / "sram_county_features.csv",
}
MEMBER_FEATURES = PROCESSED_DIR / "member_model_features.csv"
MEMBER_SDOH = PROCESSED_DIR / "member_sdoh_features.csv"
COUNTY_PRIORITY_CACHE = PROCESSED_DIR / "county_population_priorities.json"
MODEL_ARTIFACT = PROCESSED_DIR / "models" / "member_risk_model.pkl"
_FIPS_PATTERN = re.compile(r"\d{1,5}")

# A signal is actionable when it is at/above the current population median.
# priority_score = mean(severity percentile) * prevalence * affected count.
# This ranks population burden; it does not assert a clinical causal effect.
ACTIONABLE_SIGNALS = (
    ("economic_stability", "poverty_pct", "Prioritize economic stability outreach", False),
    ("economic_stability", "unemployment_pct", "Prioritize employment-support outreach", False),
    ("healthcare_access", "uninsured_pct", "Prioritize insurance-access outreach", False),
    ("housing", "housing_rent_35_plus_pct", "Prioritize housing-cost burden outreach", False),
    ("transportation", "housing_no_vehicle_pct", "Prioritize transportation-access outreach", False),
    ("digital_access", "digital_with_broadband_pct", "Prioritize digital-access outreach", True),
)


def normalize_county_fips(county_fips: str) -> str:
    value = str(county_fips).strip()
    if not _FIPS_PATTERN.fullmatch(value):
        raise ValueError("county_fips must contain one to five digits")
    return value.zfill(5)


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _canonical_fips(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(5) if re.fullmatch(r"\d{1,5}", text) else None


def _read_artifact(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required county artifact is missing: {path}")
    frame = pd.read_csv(path, dtype={"county_fips": "string"})
    if "county_fips" not in frame:
        raise ValueError(f"{path.name} has no county_fips column")
    records: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        county_fips = _canonical_fips(record.pop("county_fips"))
        if county_fips is None:
            continue
        if county_fips in records:
            raise ValueError(f"{path.name} contains duplicate county_fips: {county_fips}")
        records[county_fips] = {key: _clean_value(value) for key, value in record.items()}
    return records


@lru_cache(maxsize=1)
def _county_artifacts() -> dict[str, dict[str, dict[str, Any]]]:
    return {name: _read_artifact(path) for name, path in COUNTY_ARTIFACTS.items()}


def _metadata(sources: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    for source in (sources["county_features"], sources["sdoh_features"]):
        if source and all(source.get(key) for key in ("county_name", "state_abbr", "state_name")):
            return {key: source[key] for key in ("county_name", "state_abbr", "state_name")}
    return {"county_name": None, "state_abbr": None, "state_name": None}


def _runtime_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    members = pd.read_csv(MEMBER_FEATURES, dtype={"member_id": "string"})
    sdoh = pd.read_csv(MEMBER_SDOH, dtype={"member_id": "string"})
    for name, frame in (("member features", members), ("member SDOH", sdoh)):
        if "member_id" not in frame or not frame["member_id"].is_unique:
            raise ValueError(f"Current {name} artifact must contain unique member_id values")
    if "county_fips" not in members:
        raise ValueError("Current member features lack county_fips")
    members = members.copy()
    members["county_fips"] = members["county_fips"].map(_canonical_fips)
    return members, sdoh


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (MEMBER_FEATURES, MEMBER_SDOH, MODEL_ARTIFACT):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}".encode())
    return digest.hexdigest()


def _risk_band(mean_risk: float) -> str:
    if mean_risk >= 0.80:
        return "Very High"
    if mean_risk >= 0.60:
        return "High"
    if mean_risk >= 0.40:
        return "Moderate"
    if mean_risk >= 0.20:
        return "Low"
    return "Very Low"


def _population_snapshot() -> dict[str, Any]:
    members, sdoh = _runtime_frames()
    # Match the existing member /risk endpoint: score every current member
    # with the unchanged persisted model. member_risk_scores.csv is not used
    # because it covers only 108 rows and is not the current 1,279-member view.
    from src.modeling.train_member_risk_model import load_artifacts, predict_member_risk

    artifact = load_artifacts()
    risks = pd.DataFrame(
        predict_member_risk(row, artifact=artifact)
        for row in members.to_dict(orient="records")
    )
    merged = members.merge(
        risks[["member_id", "risk_probability", "risk_band"]],
        on="member_id", how="left", validate="one_to_one"
    )
    merged["risk_probability"] = pd.to_numeric(merged["risk_probability"], errors="coerce")
    merged.loc[~merged["risk_probability"].between(0, 1), "risk_probability"] = pd.NA
    valid_members = merged[merged["county_fips"].notna()].copy()
    by_county: dict[str, dict[str, Any]] = {}
    for fips, group in valid_members.groupby("county_fips", sort=True):
        valid_risk = group.dropna(subset=["risk_probability"])
        bands = Counter(valid_risk.get("risk_band", pd.Series(dtype="string")).dropna().astype(str))
        high_count = sum(bands[band] for band in ("High", "Very High"))
        count, risk_count = len(group), len(valid_risk)
        by_county[fips] = {
            "member_count": count, "risk_member_count": risk_count,
            "risk_coverage": risk_count / count if count else 0.0,
            "mean_risk": float(valid_risk["risk_probability"].mean()) if risk_count else None,
            "median_risk": float(valid_risk["risk_probability"].median()) if risk_count else None,
            "high_risk_count": int(bands["High"]), "very_high_risk_count": int(bands["Very High"]),
            "high_or_very_high_count": int(high_count),
            "high_or_very_high_percentage": high_count / risk_count if risk_count else None,
            "risk_band": _risk_band(float(valid_risk["risk_probability"].mean())) if risk_count else None,
            "risk_distribution": dict(sorted(bands.items())),
        }
    return {"members": valid_members, "sdoh": sdoh, "by_county": by_county}


def _priority_records(snapshot: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    members, sdoh = snapshot["members"], snapshot["sdoh"]
    # ``member_sdoh_features.csv`` also carries geography.  Preserve the
    # canonical, normalized county_fips from current member features rather
    # than allowing pandas to suffix it to county_fips_x/county_fips_y.
    sdoh_columns = [column for column in sdoh if column not in {"county_fips"}]
    joined = members[["member_id", "county_fips"]].merge(
        sdoh[sdoh_columns], on="member_id", how="left", validate="one_to_one"
    )
    if "county_fips" not in joined:
        raise ValueError("Canonical county_fips was lost while joining current member SDOH data")
    priorities: dict[str, list[dict[str, Any]]] = {}
    for domain, field, intervention, inverse in ACTIONABLE_SIGNALS:
        if field not in joined:
            continue
        values = pd.to_numeric(joined[field], errors="coerce")
        severity_values = -values if inverse else values
        valid = severity_values.dropna()
        if valid.empty:
            continue
        threshold = float(valid.median())
        percentile = severity_values.rank(pct=True)
        for fips, index in joined.groupby("county_fips", sort=True).groups.items():
            county_values, county_percentile = severity_values.loc[index], percentile.loc[index]
            affected = county_values[county_values >= threshold]
            coverage = int(county_values.notna().sum())
            if not len(affected) or not coverage:
                continue
            affected_count = int(len(affected))
            prevalence, severity = affected_count / coverage, float(county_percentile.loc[affected.index].mean())
            source_threshold = -threshold if inverse else threshold
            priorities.setdefault(fips, []).append({
                "intervention": intervention, "domain": domain,
                "affected_member_count": affected_count, "prevalence": prevalence,
                "supporting_factors": [{"field": field, "population_median": source_threshold, "direction": "below median" if inverse else "at or above median"}],
                "rationale": f"{affected_count} of {coverage} members meet the current-population threshold for {field}.",
                "source_fields_used": [field, "county_fips", "member_id"],
                "priority_score": severity * prevalence * affected_count,
                "member_data_coverage": coverage / len(index),
            })
    for records in priorities.values():
        records.sort(key=lambda item: (-item["priority_score"], item["intervention"]))
        for rank, record in enumerate(records, 1):
            record["rank"] = rank
            record["priority"] = "High" if rank == 1 else "Moderate"
    return priorities


@lru_cache(maxsize=1)
def _current_population_data() -> dict[str, Any]:
    fingerprint = _fingerprint()
    if COUNTY_PRIORITY_CACHE.exists():
        try:
            cached = json.loads(COUNTY_PRIORITY_CACHE.read_text(encoding="utf-8"))
            if cached.get("input_fingerprint") == fingerprint:
                return cached["data"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    snapshot = _population_snapshot()
    data = {"risk": snapshot["by_county"], "recommendations": _priority_records(snapshot)}
    COUNTY_PRIORITY_CACHE.write_text(json.dumps({
        "methodology": "Current member-risk aggregation and population SDOH prioritization; not member reasoner output.",
        "input_fingerprint": fingerprint, "data": data,
    }, indent=2), encoding="utf-8")
    return data


def _detail(county_fips: str) -> dict[str, Any] | None:
    population = _current_population_data()
    risk = population["risk"].get(county_fips)
    if risk is None:
        return None
    artifacts = _county_artifacts()
    sources = {name: records.get(county_fips) for name, records in artifacts.items()}
    metadata = _metadata(sources)
    if not all(metadata.values()):
        return None
    present = [name for name, record in sources.items() if record is not None]
    return {
        "county_fips": county_fips,
        **metadata,
        "data_sources": present,
        "recommendation_available": bool(population["recommendations"].get(county_fips)),
        **risk,
        **sources,
    }


def list_counties() -> list[dict[str, Any]]:
    population = _current_population_data()
    records = []
    for fips in sorted(population["risk"]):
        detail = _detail(fips)
        if detail is not None:
            records.append({key: detail[key] for key in (
                "county_fips", "county_name", "state_abbr", "state_name", "member_count",
                "risk_member_count", "risk_coverage", "data_sources",
            )} | {"recommendation_available": bool(population["recommendations"].get(fips))})
    return records


def get_county_locations() -> list[dict[str, Any]]:
    """Aggregate usable real member coordinates by the canonical county FIPS."""
    members, _ = _runtime_frames()
    coordinates = members.loc[:, ["county_fips", "lat", "lon"]].copy()
    coordinates["lat"] = pd.to_numeric(coordinates["lat"], errors="coerce")
    coordinates["lon"] = pd.to_numeric(coordinates["lon"], errors="coerce")
    coordinates = coordinates.dropna(subset=["county_fips", "lat", "lon"])
    grouped = coordinates.groupby("county_fips", sort=True).agg(lat=("lat", "mean"), lon=("lon", "mean"), mapped_member_count=("lat", "size"))
    return [{"county_fips": fips, "lat": float(row.lat), "lon": float(row.lon), "mapped_member_count": int(row.mapped_member_count)} for fips, row in grouped.iterrows()]


def get_county(county_fips: str) -> dict[str, Any] | None:
    return _detail(normalize_county_fips(county_fips))


def get_county_recommendations(county_fips: str) -> dict[str, Any] | None:
    fips = normalize_county_fips(county_fips)
    if _detail(fips) is None:
        return None
    records = _current_population_data()["recommendations"].get(fips, [])
    return {"county_fips": fips, "methodology": "Population-level prioritization from current member SDOH fields; it is not a member reasoner recommendation.", "recommendations": records}
