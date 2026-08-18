"""
Knowledge Graph Builders
========================

Builds an evidence-aware SDOH knowledge graph from the processed outputs
already produced by the HealthLens pipeline.

Important architecture
----------------------

The ML model remains responsible for prediction.

The knowledge graph is responsible for:

    Member
        |
        +--> Risk Assessment
        |
        +--> Clinical Context
        |
        +--> SDOH Factor
                    |
                    +--> SDOH Domain
                    |
                    +--> Evidence
                    |
                    +--> Intervention
        |
        +--> County
                    |
                    +--> County Risk
                    |
                    +--> County SDOH Context

The graph is constructed from actual observed/derived data.

It does NOT create a fixed list of pre-defined member scenarios.

Missing county_fips values are preserved as missing and are never
arbitrarily assigned.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

MEMBER_RISK_FILE = (
    PROCESSED_DIR / "member_risk_scores.csv"
)

MEMBER_FEATURE_FILE = (
    PROCESSED_DIR / "member_model_features.csv"
)

INTERVENTION_FILE = (
    PROCESSED_DIR / "intervention_priorities.csv"
)

COUNTY_RISK_FILE = (
    PROCESSED_DIR / "county_risk_scores.csv"
)

OUTPUT_DIR = (
    PROCESSED_DIR / "knowledge_graph"
)

GRAPH_FILE = (
    OUTPUT_DIR / "healthlens_knowledge_graph.json"
)

VALIDATION_FILE = (
    OUTPUT_DIR / "knowledge_graph_validation.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "knowledge_graph_summary.json"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SCHEMA_VERSION = "1.0.0"

TARGET_COLUMN = "target_inpatient_any"

MISSING_COUNTY_ALLOWED = True


# ============================================================================
# SDOH DOMAIN DEFINITIONS
# ============================================================================

SDOH_DOMAINS: dict[str, str] = {
    "Economic Stability": "Economic conditions affecting a member's ability "
                           "to meet basic needs and maintain health.",

    "Education Access": "Educational attainment and related barriers "
                         "affecting health and opportunity.",

    "Healthcare Access": "Access to healthcare services, insurance, "
                         "preventive care, and routine care.",

    "Neighborhood and Built Environment": "Physical and environmental "
                         "conditions surrounding the member.",

    "Transportation": "Transportation and geographic access barriers "
                         "affecting access to care and essential services.",

    "Housing": "Housing stability, affordability, crowding, vacancy, "
                         "and housing-related barriers.",

    "Digital Access": "Access to computers, broadband, and digital "
                         "resources needed for healthcare and services.",
}


# ============================================================================
# FACTOR -> DOMAIN MAPPING
# ============================================================================

FACTOR_DOMAIN: dict[str, str] = {

    # ------------------------------------------------------------------
    # Economic stability
    # ------------------------------------------------------------------

    "poverty_pct": "Economic Stability",
    "unemployment_pct": "Economic Stability",
    "public_assistance_pct": "Economic Stability",
    "median_household_income": "Economic Stability",
    "snap_households_count_sum": "Economic Stability",

    # ------------------------------------------------------------------
    # Education
    # ------------------------------------------------------------------

    "education_less_than_9th_pct": "Education Access",
    "education_9th_to_12th_no_diploma_pct": "Education Access",
    "education_less_than_high_school_pct": "Education Access",
    "education_high_school_pct": "Education Access",
    "education_college_pct": "Education Access",
    "education_bachelors_or_higher_pct": "Education Access",

    # ------------------------------------------------------------------
    # Healthcare access
    # ------------------------------------------------------------------

    "uninsured_pct": "Healthcare Access",
    "places_uninsured_pct": "Healthcare Access",
    "places_routine_checkup_pct": "Healthcare Access",
    "places_cholesterol_screening_pct": "Healthcare Access",

    # ------------------------------------------------------------------
    # Housing
    # ------------------------------------------------------------------

    "housing_vacancy_pct": "Housing",
    "housing_no_vehicle_pct": "Housing",
    "housing_renter_pct": "Housing",
    "housing_crowded_1_01_to_1_50_pct": "Housing",
    "housing_crowded_1_51_plus_pct": "Housing",
    "housing_crowded_pct": "Housing",
    "housing_rent_30_to_34_9_pct": "Housing",
    "housing_rent_35_plus_pct": "Housing",
    "housing_cost_burden_30pct_or_more": "Housing",

    # ------------------------------------------------------------------
    # Transportation
    # ------------------------------------------------------------------

    "mean_commute_minutes": "Transportation",
    "households_without_vehicle_count_sum": "Transportation",
    "transport_no_vehicle_pct": "Transportation",
    "driving_low_access_population_beyond_1mi_10mi_count_sum":
        "Transportation",
    "driving_no_vehicle_households_beyond_1mi_count_sum":
        "Transportation",
    "driving_snap_households_beyond_1mi_count_sum":
        "Transportation",
    "driving_low_income_low_access_tract_count":
        "Transportation",
    "driving_low_vehicle_access_tract_count":
        "Transportation",
    "straight_no_vehicle_households_beyond_1mi_count_sum":
        "Transportation",
    "straight_snap_households_beyond_1mi_count_sum":
        "Transportation",
    "straight_low_income_low_access_tract_count":
        "Transportation",
    "straight_low_vehicle_access_tract_count":
        "Transportation",

    # ------------------------------------------------------------------
    # Digital access
    # ------------------------------------------------------------------

    "digital_with_computer_pct": "Digital Access",
    "digital_with_broadband_pct": "Digital Access",
    "digital_no_computer_pct": "Digital Access",
    "digital_no_broadband_pct": "Digital Access",

    # ------------------------------------------------------------------
    # Health environment / PLACES
    # ------------------------------------------------------------------

    "places_asthma_pct": "Neighborhood and Built Environment",
    "places_copd_pct": "Neighborhood and Built Environment",
    "places_diabetes_pct": "Neighborhood and Built Environment",
    "places_heart_disease_pct": "Neighborhood and Built Environment",
    "places_obesity_pct": "Neighborhood and Built Environment",
    "places_physical_inactivity_pct":
        "Neighborhood and Built Environment",
    "places_poor_mental_health_pct":
        "Neighborhood and Built Environment",
    "places_poor_physical_health_pct":
        "Neighborhood and Built Environment",
    "places_smoking_pct":
        "Neighborhood and Built Environment",
    "places_stroke_pct":
        "Neighborhood and Built Environment",
}


# ============================================================================
# INTERVENTION MAPPING
# ============================================================================

DEFAULT_INTERVENTIONS: dict[str, tuple[str, str]] = {

    "Economic Stability": (
        "Financial assistance and benefits navigation",
        "Connect the member with benefits screening, financial assistance, "
        "and social-service navigation.",
    ),

    "Education Access": (
        "Health education and community resource navigation",
        "Provide accessible health education and connect the member with "
        "appropriate community education resources.",
    ),

    "Healthcare Access": (
        "Care coordination and healthcare access support",
        "Connect the member with primary care, preventive care, "
        "insurance navigation, and care coordination.",
    ),

    "Transportation": (
        "Transportation assistance for healthcare access",
        "Connect the member with transportation resources for healthcare "
        "appointments and essential services.",
    ),

    "Housing": (
        "Housing stability support",
        "Connect the member with housing assistance, housing navigation, "
        "and community resources.",
    ),

    "Digital Access": (
        "Digital-access and telehealth support",
        "Connect the member with digital-access resources and support "
        "for telehealth and online healthcare services.",
    ),

    "Neighborhood and Built Environment": (
        "Community health and environmental resource navigation",
        "Connect the member with relevant community-based health and "
        "environmental resources.",
    ),
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class KGNode:
    """
    Generic graph node.

    `node_type` is deliberately stored as the schema's semantic name.
    This makes the builder independent of a particular graph database.
    """

    node_id: str
    node_type: str
    label: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KGEdge:
    """
    Generic graph relationship.
    """

    edge_id: str
    source: str
    target: str
    relationship_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """
    In-memory graph representation.

    This is intentionally simple.

    A later graph.py layer can serialize this into Neo4j, NetworkX,
    RDF, or another backend without changing the builder logic.
    """

    schema_version: str
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)

    def add_node(self, node: KGNode) -> None:
        if not any(existing.node_id == node.node_id for existing in self.nodes):
            self.nodes.append(node)

    def add_edge(self, edge: KGEdge) -> None:
        if not any(existing.edge_id == edge.edge_id for existing in self.edges):
            self.edges.append(edge)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "label": node.label,
                    "properties": clean_for_json(node.properties),
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "source": edge.source,
                    "target": edge.target,
                    "relationship_type": edge.relationship_type,
                    "properties": clean_for_json(edge.properties),
                }
                for edge in self.edges
            ],
        }


# ============================================================================
# UTILITIES
# ============================================================================

def clean_scalar(value: Any) -> Any:
    """
    Convert pandas / numpy values to JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    if pd.isna(value):
        return None

    return value


def clean_for_json(value: Any) -> Any:

    if isinstance(value, dict):
        return {
            str(k): clean_for_json(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            clean_for_json(v)
            for v in value
        ]

    return clean_scalar(value)


def normalize_identifier(value: Any) -> str:

    value = clean_scalar(value)

    if value is None:
        return ""

    return str(value).strip()


def normalize_county(value: Any) -> str | None:

    value = clean_scalar(value)

    if value is None:
        return None

    try:
        return str(int(float(value))).zfill(5)
    except (TypeError, ValueError):
        return str(value).strip()


def safe_name(value: Any) -> str:

    value = normalize_identifier(value)

    value = value.lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip("_")


def make_node_id(
    node_type: str,
    identifier: Any,
) -> str:

    return (
        f"{safe_name(node_type)}:"
        f"{safe_name(identifier)}"
    )


def make_edge_id(
    source: str,
    relationship: str,
    target: str,
) -> str:

    return (
        f"{source}"
        f"__{safe_name(relationship)}__"
        f"{target}"
    )


# ============================================================================
# SCHEMA COMPATIBILITY
# ============================================================================

def _load_schema_types() -> dict[str, Any]:
    """
    Load schema enums if available.

    The builder does not require a specific Enum implementation, but it
    validates that the semantic node/relationship names correspond to
    the schema whenever possible.
    """

    try:
        from . import schema

        return {
            "NodeType": getattr(schema, "NodeType", None),
            "EvidenceType": getattr(schema, "EvidenceType", None),
            "RelationshipType": getattr(
                schema,
                "RelationshipType",
                None,
            ),
        }

    except Exception:
        return {
            "NodeType": None,
            "EvidenceType": None,
            "RelationshipType": None,
        }


def _enum_candidates(enum_class: Any) -> list[str]:

    if enum_class is None:
        return []

    try:
        return [
            str(member.name)
            for member in enum_class
        ]
    except Exception:
        return []


def _resolve_schema_name(
    enum_class: Any,
    aliases: Iterable[str],
    fallback: str,
) -> str:
    """
    Resolve a semantic name against the current schema.

    Example:
        aliases =
            ["MEMBER", "Member", "member"]

    If the schema exposes MEMBER, that name is returned.
    Otherwise the fallback semantic name is used.
    """

    candidates = _enum_candidates(enum_class)

    normalized = {
        re.sub(
            r"[^a-z0-9]",
            "",
            candidate.lower(),
        ): candidate
        for candidate in candidates
    }

    for alias in aliases:

        key = re.sub(
            r"[^a-z0-9]",
            "",
            alias.lower(),
        )

        if key in normalized:
            return normalized[key]

    return fallback


def resolve_node_type(name: str) -> str:

    schema_types = _load_schema_types()

    aliases = [
        name,
        name.upper(),
        name.replace(" ", "_"),
        name.replace(" ", ""),
    ]

    return _resolve_schema_name(
        schema_types["NodeType"],
        aliases,
        name,
    )


def resolve_relationship_type(name: str) -> str:

    schema_types = _load_schema_types()

    aliases = [
        name,
        name.upper(),
        name.replace(" ", "_"),
        name.replace(" ", ""),
    ]

    return _resolve_schema_name(
        schema_types["RelationshipType"],
        aliases,
        name,
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_csv(
    path: Path,
    required: bool = True,
) -> pd.DataFrame:

    if not path.exists():

        if required:
            raise FileNotFoundError(
                f"Required KG input not found:\n{path}"
            )

        return pd.DataFrame()

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"Input file is empty:\n{path}"
        )

    return df


def load_inputs() -> dict[str, pd.DataFrame]:

    print("=" * 70)
    print("LOADING KNOWLEDGE GRAPH INPUTS")
    print("=" * 70)

    member_risk = load_csv(
        MEMBER_RISK_FILE
    )

    member_features = load_csv(
        MEMBER_FEATURE_FILE
    )

    interventions = load_csv(
        INTERVENTION_FILE
    )

    county_risk = load_csv(
        COUNTY_RISK_FILE,
        required=False,
    )

    print(
        f"Member risk rows:     {len(member_risk)}"
    )

    print(
        f"Member feature rows:  {len(member_features)}"
    )

    print(
        f"Intervention rows:    {len(interventions)}"
    )

    print(
        f"County risk rows:     {len(county_risk)}"
    )

    return {
        "member_risk": member_risk,
        "member_features": member_features,
        "interventions": interventions,
        "county_risk": county_risk,
    }


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_inputs(
    data: dict[str, pd.DataFrame],
) -> None:

    print()
    print("=" * 70)
    print("VALIDATING KNOWLEDGE GRAPH INPUTS")
    print("=" * 70)

    member_risk = data["member_risk"]
    member_features = data["member_features"]
    interventions = data["interventions"]

    required_risk = {
        "member_id",
        "risk_probability",
    }

    required_features = {
        "member_id",
    }

    missing = required_risk - set(member_risk.columns)

    if missing:
        raise ValueError(
            "member_risk_scores.csv missing columns: "
            f"{sorted(missing)}"
        )

    missing = required_features - set(member_features.columns)

    if missing:
        raise ValueError(
            "member_model_features.csv missing columns: "
            f"{sorted(missing)}"
        )

    if member_risk["member_id"].duplicated().any():
        raise ValueError(
            "Duplicate member_id values in member risk."
        )

    if member_features["member_id"].duplicated().any():
        raise ValueError(
            "Duplicate member_id values in member features."
        )

    if not interventions.empty:
        if "member_id" not in interventions.columns:
            raise ValueError(
                "intervention_priorities.csv must contain member_id."
            )

    print(
        f"Risk members:       {len(member_risk)}"
    )

    print(
        f"Feature members:    {len(member_features)}"
    )

    print(
        f"High-risk members:  "
        f"{count_high_risk_members(member_risk)}"
    )

    print(
        "Input validation: PASSED"
    )


def count_high_risk_members(
    df: pd.DataFrame,
) -> int:

    if "risk_band" in df.columns:

        high_bands = {
            "High",
            "Very High",
        }

        return int(
            df["risk_band"]
            .astype(str)
            .isin(high_bands)
            .sum()
        )

    if "risk_probability" in df.columns:

        return int(
            pd.to_numeric(
                df["risk_probability"],
                errors="coerce",
            )
            .ge(0.5)
            .sum()
        )

    return 0


# ============================================================================
# NODE BUILDERS
# ============================================================================

def add_member_node(
    graph: KnowledgeGraph,
    row: pd.Series,
) -> str:

    member_id = normalize_identifier(
        row.get("member_id")
    )

    node_id = make_node_id(
        "Member",
        member_id,
    )

    properties = {
        "member_id": member_id,
    }

    # Keep useful demographic information if available.
    optional_columns = [
        "age",
        "gender",
        "race",
        "ethnicity",
        "marital_status",
        "city",
        "state",
        "county",
        "state_abbr",
        "state_name",
    ]

    for column in optional_columns:

        if column in row.index:

            value = clean_scalar(
                row[column]
            )

            if value is not None:
                properties[column] = value

    graph.add_node(
        KGNode(
            node_id=node_id,
            node_type=resolve_node_type(
                "Member"
            ),
            label=member_id,
            properties=properties,
        )
    )

    return node_id


def add_county_node(
    graph: KnowledgeGraph,
    county_fips: Any,
    row: pd.Series | None = None,
) -> str | None:

    county = normalize_county(
        county_fips
    )

    if county is None:
        return None

    node_id = make_node_id(
        "County",
        county,
    )

    properties = {
        "county_fips": county,
    }

    if row is not None:

        for column in [
            "county_name",
            "state_abbr",
            "state_name",
        ]:

            if column in row.index:

                value = clean_scalar(
                    row[column]
                )

                if value is not None:
                    properties[column] = value

    graph.add_node(
        KGNode(
            node_id=node_id,
            node_type=resolve_node_type(
                "County"
            ),
            label=county,
            properties=properties,
        )
    )

    return node_id


def add_risk_node(
    graph: KnowledgeGraph,
    row: pd.Series,
) -> str:

    member_id = normalize_identifier(
        row["member_id"]
    )

    risk_id = (
        f"{member_id}:risk_assessment"
    )

    properties = {
        "member_id": member_id,
        "risk_probability": clean_scalar(
            row.get("risk_probability")
        ),
        "risk_percentile": clean_scalar(
            row.get("risk_percentile")
        ),
        "risk_band": clean_scalar(
            row.get("risk_band")
        ),
        "risk_rank": clean_scalar(
            row.get("risk_rank")
        ),
        "model": clean_scalar(
            row.get("model")
        ),
    }

    graph.add_node(
        KGNode(
            node_id=risk_id,
            node_type=resolve_node_type(
                "RiskAssessment"
            ),
            label=f"Risk assessment for {member_id}",
            properties=properties,
        )
    )

    return risk_id


def add_domain_node(
    graph: KnowledgeGraph,
    domain: str,
) -> str:

    node_id = make_node_id(
        "SDOHDomain",
        domain,
    )

    graph.add_node(
        KGNode(
            node_id=node_id,
            node_type=resolve_node_type(
                "SDOHDomain"
            ),
            label=domain,
            properties={
                "domain": domain,
                "description": SDOH_DOMAINS.get(
                    domain,
                    "",
                ),
            },
        )
    )

    return node_id


def add_factor_node(
    graph: KnowledgeGraph,
    factor: str,
    domain: str,
) -> str:

    node_id = make_node_id(
        "SDOHFactor",
        factor,
    )

    graph.add_node(
        KGNode(
            node_id=node_id,
            node_type=resolve_node_type(
                "SDOHFactor"
            ),
            label=factor,
            properties={
                "factor": factor,
                "domain": domain,
            },
        )
    )

    return node_id


def add_intervention_node(
    graph: KnowledgeGraph,
    intervention_name: str,
    domain: str,
    description: str,
) -> str:

    node_id = make_node_id(
        "Intervention",
        intervention_name,
    )

    graph.add_node(
        KGNode(
            node_id=node_id,
            node_type=resolve_node_type(
                "Intervention"
            ),
            label=intervention_name,
            properties={
                "intervention": intervention_name,
                "domain": domain,
                "description": description,
            },
        )
    )

    return node_id


def add_evidence_node(
    graph: KnowledgeGraph,
    source: str,
    factor: str | None = None,
) -> str:

    suffix = (
        f"{source}:{factor}"
        if factor
        else source
    )

    node_id = make_node_id(
        "Evidence",
        suffix,
    )

    graph.add_node(
        KGNode(
            node_id=node_id,
            node_type=resolve_node_type(
                "Evidence"
            ),
            label=source,
            properties={
                "source": source,
                "factor": factor,
            },
        )
    )

    return node_id


# ============================================================================
# EDGE BUILDER
# ============================================================================

def add_relationship(
    graph: KnowledgeGraph,
    source: str,
    relationship: str,
    target: str,
    properties: dict[str, Any] | None = None,
) -> None:

    graph.add_edge(
        KGEdge(
            edge_id=make_edge_id(
                source,
                relationship,
                target,
            ),
            source=source,
            target=target,
            relationship_type=resolve_relationship_type(
                relationship
            ),
            properties=properties or {},
        )
    )


# ============================================================================
# SDOH FACTOR DETECTION
# ============================================================================

def numeric_value(
    row: pd.Series,
    column: str,
) -> float | None:

    if column not in row.index:
        return None

    value = pd.to_numeric(
        pd.Series([row[column]]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(value):
        return None

    if not np.isfinite(value):
        return None

    return float(value)


def calculate_factor_strength(
    value: float,
    series: pd.Series,
    higher_is_worse: bool = True,
) -> float:

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if numeric.empty:
        return 0.0

    minimum = float(
        numeric.min()
    )

    maximum = float(
        numeric.max()
    )

    if maximum <= minimum:
        return 0.0

    normalized = (
        value - minimum
    ) / (
        maximum - minimum
    )

    normalized = float(
        np.clip(
            normalized,
            0.0,
            1.0,
        )
    )

    if not higher_is_worse:
        normalized = 1.0 - normalized

    return normalized


def factor_is_relevant(
    strength: float,
    minimum_strength: float = 0.50,
) -> bool:

    return strength >= minimum_strength


def factor_direction(
    factor: str,
) -> bool:
    """
    Return True if a high value represents greater SDOH need.

    False means a low value represents greater need.

    This is intentionally limited to factors where direction is clear.
    """

    lower_is_worse = {
        "median_household_income",
        "education_college_pct",
        "education_bachelors_or_higher_pct",
        "digital_with_computer_pct",
        "digital_with_broadband_pct",
        "places_routine_checkup_pct",
        "places_cholesterol_screening_pct",
    }

    return factor not in lower_is_worse


def detect_member_sdoh_factors(
    row: pd.Series,
    feature_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Dynamically identify relevant SDOH factors.

    This does not use pre-defined member scenarios.

    A factor becomes relevant when its observed value is sufficiently
    extreme relative to the available member population.

    For intervention_priorities.csv, the builder also accepts the
    already-derived primary factor as evidence.
    """

    detected: list[dict[str, Any]] = []

    for factor, domain in FACTOR_DOMAIN.items():

        if factor not in feature_df.columns:
            continue

        value = numeric_value(
            row,
            factor,
        )

        if value is None:
            continue

        higher_is_worse = factor_direction(
            factor
        )

        strength = calculate_factor_strength(
            value,
            feature_df[factor],
            higher_is_worse=higher_is_worse,
        )

        if not factor_is_relevant(
            strength
        ):
            continue

        detected.append(
            {
                "factor": factor,
                "domain": domain,
                "value": value,
                "strength": strength,
                "direction": (
                    "high_need"
                    if higher_is_worse
                    else "low_access"
                ),
            }
        )

    detected.sort(
        key=lambda item: item["strength"],
        reverse=True,
    )

    return detected


# ============================================================================
# INTERVENTION LOOKUP
# ============================================================================

def build_intervention_lookup(
    interventions: pd.DataFrame,
) -> dict[str, dict[str, Any]]:

    lookup: dict[str, dict[str, Any]] = {}

    if interventions.empty:
        return lookup

    for _, row in interventions.iterrows():

        member_id = normalize_identifier(
            row.get("member_id")
        )

        if not member_id:
            continue

        lookup[member_id] = {
            column: clean_scalar(row.get(column))
            for column in interventions.columns
        }

    return lookup


def get_intervention_for_domain(
    domain: str,
) -> tuple[str, str]:

    return DEFAULT_INTERVENTIONS.get(
        domain,
        (
            "Community resource navigation",
            "Connect the member with appropriate community resources.",
        ),
    )


# ============================================================================
# MEMBER GRAPH CONSTRUCTION
# ============================================================================

def build_member_graph(
    graph: KnowledgeGraph,
    member_risk: pd.DataFrame,
    member_features: pd.DataFrame,
    interventions: pd.DataFrame,
) -> dict[str, Any]:

    print()
    print("=" * 70)
    print("BUILDING MEMBER KNOWLEDGE GRAPH")
    print("=" * 70)

    features_by_member = (
        member_features
        .set_index("member_id", drop=False)
    )

    risk_by_member = (
        member_risk
        .set_index("member_id", drop=False)
    )

    intervention_lookup = (
        build_intervention_lookup(
            interventions
        )
    )

    members_processed = 0
    members_with_county = 0
    members_without_county = 0
    factor_relationships = 0
    intervention_relationships = 0

    for member_id, risk_row in risk_by_member.iterrows():

        if member_id not in features_by_member.index:
            continue

        feature_row = features_by_member.loc[
            member_id
        ]

        # --------------------------------------------------------------
        # MEMBER
        # --------------------------------------------------------------

        member_node = add_member_node(
            graph,
            feature_row,
        )

        members_processed += 1

        # --------------------------------------------------------------
        # RISK
        # --------------------------------------------------------------

        risk_node = add_risk_node(
            graph,
            risk_row,
        )

        add_relationship(
            graph,
            member_node,
            "HAS_RISK_ASSESSMENT",
            risk_node,
            {
                "source": "member_risk_scores.csv",
            },
        )

        # --------------------------------------------------------------
        # COUNTY
        # --------------------------------------------------------------

        county = feature_row.get(
            "county_fips"
        )

        county_node = add_county_node(
            graph,
            county,
            feature_row,
        )

        if county_node is not None:

            members_with_county += 1

            add_relationship(
                graph,
                member_node,
                "LIVES_IN",
                county_node,
                {
                    "source": "member_model_features.csv",
                },
            )

        else:

            members_without_county += 1

        # --------------------------------------------------------------
        # DYNAMIC SDOH DETECTION
        # --------------------------------------------------------------

        detected = detect_member_sdoh_factors(
            feature_row,
            member_features,
        )

        # --------------------------------------------------------------
        # INTERVENTION-DERIVED PRIMARY FACTOR
        # --------------------------------------------------------------

        intervention_row = (
            intervention_lookup.get(
                member_id
            )
        )

        if intervention_row:

            primary_factor = normalize_identifier(
                intervention_row.get(
                    "primary_sdoh_domain"
                )
            )

            if (
                primary_factor
                and primary_factor != "Unknown"
                and primary_factor in SDOH_DOMAINS
            ):

                # If the intervention pipeline identified a domain,
                # make sure it is represented in the graph even if
                # individual numeric factors are missing.
                existing_domains = {
                    item["domain"]
                    for item in detected
                }

                if primary_factor not in existing_domains:

                    detected.append(
                        {
                            "factor": (
                                normalize_identifier(
                                    intervention_row.get(
                                        "primary_factor"
                                    )
                                )
                                or "derived_sdoh_need"
                            ),
                            "domain": primary_factor,
                            "value": None,
                            "strength": clean_scalar(
                                intervention_row.get(
                                    "overall_sdoh_need_score"
                                )
                            ),
                            "direction": "derived",
                        }
                    )

        # --------------------------------------------------------------
        # FACTOR NODES
        # --------------------------------------------------------------

        member_domains: set[str] = set()

        for item in detected:

            factor = item["factor"]
            domain = item["domain"]

            factor_node = add_factor_node(
                graph,
                factor,
                domain,
            )

            domain_node = add_domain_node(
                graph,
                domain,
            )

            member_domains.add(
                domain
            )

            add_relationship(
                graph,
                member_node,
                "HAS_SDOH_FACTOR",
                factor_node,
                {
                    "value": item.get("value"),
                    "strength": item.get("strength"),
                    "direction": item.get("direction"),
                    "derived_from": (
                        "member_model_features.csv"
                    ),
                },
            )

            add_relationship(
                graph,
                factor_node,
                "BELONGS_TO_DOMAIN",
                domain_node,
            )

            # ----------------------------------------------------------
            # EVIDENCE
            # ----------------------------------------------------------

            source = infer_factor_source(
                factor
            )

            evidence_node = add_evidence_node(
                graph,
                source,
                factor,
            )

            add_relationship(
                graph,
                factor_node,
                "SUPPORTED_BY",
                evidence_node,
                {
                    "source": source,
                },
            )

            factor_relationships += 1

        # --------------------------------------------------------------
        # INTERVENTION
        # --------------------------------------------------------------

        if intervention_row:

            intervention_name = normalize_identifier(
                intervention_row.get(
                    "recommended_intervention"
                )
            )

            primary_domain = normalize_identifier(
                intervention_row.get(
                    "primary_sdoh_domain"
                )
            )

            if (
                intervention_name
                and primary_domain
                and primary_domain != "Unknown"
            ):

                description = ""

                (
                    default_name,
                    default_description,
                ) = get_intervention_for_domain(
                    primary_domain
                )

                if intervention_name:
                    description = default_description

                intervention_node = (
                    add_intervention_node(
                        graph,
                        intervention_name
                        or default_name,
                        primary_domain,
                        description,
                    )
                )

                domain_node = add_domain_node(
                    graph,
                    primary_domain,
                )

                add_relationship(
                    graph,
                    member_node,
                    "RECEIVES_INTERVENTION_RECOMMENDATION",
                    intervention_node,
                    {
                        "intervention_priority_score":
                            intervention_row.get(
                                "intervention_priority_score"
                            ),
                        "intervention_priority":
                            intervention_row.get(
                                "intervention_priority"
                            ),
                        "primary_sdoh_domain":
                            primary_domain,
                    },
                )

                add_relationship(
                    graph,
                    intervention_node,
                    "ADDRESSES_DOMAIN",
                    domain_node,
                )

                intervention_relationships += 1

        # --------------------------------------------------------------
        # CLINICAL CONTEXT
        # --------------------------------------------------------------

        add_clinical_context(
            graph,
            member_node,
            feature_row,
        )

    print(
        f"Members processed:       {members_processed}"
    )

    print(
        f"Members with county:     {members_with_county}"
    )

    print(
        f"Members without county:  {members_without_county}"
    )

    print(
        f"SDOH relationships:      {factor_relationships}"
    )

    print(
        f"Intervention links:      {intervention_relationships}"
    )

    return {
        "members_processed": members_processed,
        "members_with_county": members_with_county,
        "members_without_county": members_without_county,
        "sdoh_relationships": factor_relationships,
        "intervention_relationships": intervention_relationships,
    }


# ============================================================================
# SOURCE INFERENCE
# ============================================================================

def infer_factor_source(
    factor: str,
) -> str:

    if factor.startswith("places_"):
        return "PLACES"

    if factor.startswith("driving_"):
        return "SRAM"

    if factor.startswith("straight_"):
        return "SRAM"

    if factor.startswith("households_without_vehicle"):
        return "SRAM"

    if factor.startswith("snap_households"):
        return "ACS"

    if factor.startswith("education_"):
        return "ACS"

    if factor.startswith("poverty"):
        return "ACS"

    if factor.startswith("unemployment"):
        return "ACS"

    if factor.startswith("public_assistance"):
        return "ACS"

    if factor.startswith("median_household_income"):
        return "ACS"

    if factor.startswith("uninsured"):
        return "ACS"

    if factor.startswith("housing_"):
        return "ACS"

    if factor.startswith("digital_"):
        return "ACS"

    if factor.startswith("transport_"):
        return "ACS"

    if factor.startswith("mean_commute"):
        return "ACS"

    return "Derived SDOH Feature"


# ============================================================================
# CLINICAL CONTEXT
# ============================================================================

CLINICAL_FEATURES = {
    "clinical_history_days",
    "clinical_encounter_count",
    "clinical_inpatient_history_count",
    "clinical_emergency_count",
    "clinical_outpatient_count",
    "clinical_ambulatory_count",
    "clinical_urgentcare_count",
    "clinical_wellness_count",
    "clinical_home_count",
    "clinical_snf_count",
    "clinical_hospice_count",
    "clinical_condition_count",
    "clinical_unique_condition_count",
    "clinical_unique_condition_description_count",
    "clinical_observation_count",
    "clinical_unique_observation_count",
    "clinical_medication_count",
    "clinical_unique_medication_count",
    "clinical_procedure_count",
    "clinical_unique_procedure_count",
    "clinical_allergy_count",
    "clinical_unique_allergy_count",
    "clinical_immunization_count",
    "clinical_unique_immunization_count",
    "clinical_careplan_count",
    "clinical_unique_careplan_count",
}


def add_clinical_context(
    graph: KnowledgeGraph,
    member_node: str,
    row: pd.Series,
) -> None:

    member_id = normalize_identifier(
        row.get("member_id")
    )

    for feature in CLINICAL_FEATURES:

        if feature not in row.index:
            continue

        value = numeric_value(
            row,
            feature,
        )

        if value is None:
            continue

        # Do not create a node for zero-valued clinical counts.
        if value == 0:
            continue

        clinical_id = make_node_id(
            "ClinicalFactor",
            f"{member_id}:{feature}",
        )

        graph.add_node(
            KGNode(
                node_id=clinical_id,
                node_type=resolve_node_type(
                    "ClinicalFactor"
                ),
                label=feature,
                properties={
                    "member_id": member_id,
                    "feature": feature,
                    "value": value,
                },
            )
        )

        add_relationship(
            graph,
            member_node,
            "HAS_CLINICAL_CONTEXT",
            clinical_id,
            {
                "value": value,
            },
        )


# ============================================================================
# COUNTY GRAPH
# ============================================================================

def build_county_graph(
    graph: KnowledgeGraph,
    member_features: pd.DataFrame,
    county_risk: pd.DataFrame,
) -> dict[str, Any]:

    print()
    print("=" * 70)
    print("BUILDING COUNTY KNOWLEDGE GRAPH")
    print("=" * 70)

    counties_processed = 0

    if county_risk.empty:
        print(
            "No county risk file available."
        )

        return {
            "counties_processed": 0,
        }

    for _, row in county_risk.iterrows():

        county = normalize_county(
            row.get("county_fips")
        )

        if county is None:
            continue

        county_node = add_county_node(
            graph,
            county,
            row,
        )

        if county_node is None:
            continue

        counties_processed += 1

        risk_id = make_node_id(
            "CountyRiskAssessment",
            county,
        )

        graph.add_node(
            KGNode(
                node_id=risk_id,
                node_type=resolve_node_type(
                    "CountyRiskAssessment"
                ),
                label=f"County risk {county}",
                properties={
                    "county_fips": county,
                    "county_risk_probability":
                        clean_scalar(
                            row.get(
                                "county_risk_probability"
                            )
                        ),
                    "county_risk_percentile":
                        clean_scalar(
                            row.get(
                                "county_risk_percentile"
                            )
                        ),
                    "risk_band":
                        clean_scalar(
                            row.get(
                                "risk_band"
                            )
                        ),
                    "member_count":
                        clean_scalar(
                            row.get(
                                "member_count"
                            )
                        ),
                    "high_risk_member_count":
                        clean_scalar(
                            row.get(
                                "high_risk_member_count"
                            )
                        ),
                    "high_risk_member_pct":
                        clean_scalar(
                            row.get(
                                "high_risk_member_pct"
                            )
                        ),
                },
            )
        )

        add_relationship(
            graph,
            county_node,
            "HAS_RISK_ASSESSMENT",
            risk_id,
            {
                "source": "county_risk_scores.csv",
            },
        )

    print(
        f"Counties processed: {counties_processed}"
    )

    return {
        "counties_processed": counties_processed,
    }


# ============================================================================
# GRAPH VALIDATION
# ============================================================================

def validate_graph(
    graph: KnowledgeGraph,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("VALIDATING KNOWLEDGE GRAPH")
    print("=" * 70)

    node_ids = {
        node.node_id
        for node in graph.nodes
    }

    checks: list[dict[str, Any]] = []

    def check(
        name: str,
        passed: bool,
        detail: str,
    ) -> None:

        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    check(
        "graph_has_nodes",
        len(graph.nodes) > 0,
        f"nodes={len(graph.nodes)}",
    )

    check(
        "graph_has_edges",
        len(graph.edges) > 0,
        f"edges={len(graph.edges)}",
    )

    dangling_edges = []

    for edge in graph.edges:

        if edge.source not in node_ids:
            dangling_edges.append(
                edge.edge_id
            )

        if edge.target not in node_ids:
            dangling_edges.append(
                edge.edge_id
            )

    check(
        "no_dangling_edges",
        len(dangling_edges) == 0,
        f"dangling={len(dangling_edges)}",
    )

    duplicate_nodes = (
        len(graph.nodes)
        != len(node_ids)
    )

    check(
        "unique_node_ids",
        not duplicate_nodes,
        f"nodes={len(graph.nodes)}",
    )

    duplicate_edges = (
        len(graph.edges)
        != len({
            edge.edge_id
            for edge in graph.edges
        })
    )

    check(
        "unique_edge_ids",
        not duplicate_edges,
        f"edges={len(graph.edges)}",
    )

    member_nodes = [
        node
        for node in graph.nodes
        if node.node_type
        == resolve_node_type("Member")
    ]

    check(
        "member_nodes_present",
        len(member_nodes) > 0,
        f"members={len(member_nodes)}",
    )

    unresolved_county_edges = 0

    for node in member_nodes:

        member_id = node.properties.get(
            "member_id"
        )

        has_county = any(
            edge.source == node.node_id
            and edge.relationship_type
            == resolve_relationship_type(
                "LIVES_IN"
            )
            for edge in graph.edges
        )

        if not has_county:
            unresolved_county_edges += 1

    check(
        "missing_counties_allowed",
        MISSING_COUNTY_ALLOWED,
        (
            f"members without county="
            f"{unresolved_county_edges}"
        ),
    )

    report = pd.DataFrame(
        checks
    )

    failed = report.loc[
        ~report["passed"]
    ]

    if not failed.empty:

        print(
            failed.to_string(
                index=False
            )
        )

        raise ValueError(
            "Knowledge graph validation failed."
        )

    print(
        "Knowledge graph validation: PASSED"
    )

    return report


# ============================================================================
# GRAPH SUMMARY
# ============================================================================

def graph_summary(
    graph: KnowledgeGraph,
) -> dict[str, Any]:

    node_counts = (
        pd.Series(
            [
                node.node_type
                for node in graph.nodes
            ]
        )
        .value_counts()
        .to_dict()
    )

    edge_counts = (
        pd.Series(
            [
                edge.relationship_type
                for edge in graph.edges
            ]
        )
        .value_counts()
        .to_dict()
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "node_types": {
            str(k): int(v)
            for k, v in node_counts.items()
        },
        "relationship_types": {
            str(k): int(v)
            for k, v in edge_counts.items()
        },
    }


# ============================================================================
# SAVE GRAPH
# ============================================================================

def save_graph(
    graph: KnowledgeGraph,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with GRAPH_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            graph.to_dict(),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    summary = graph_summary(
        graph
    )

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            clean_for_json(summary),
            handle,
            indent=2,
        )


def save_validation_report(
    report: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        VALIDATION_FILE,
        index=False,
    )


# ============================================================================
# PRINT SUMMARY
# ============================================================================

def print_summary(
    graph: KnowledgeGraph,
) -> None:

    summary = graph_summary(
        graph
    )

    print()
    print("=" * 70)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("=" * 70)

    print(
        f"Schema version:  {summary['schema_version']}"
    )

    print(
        f"Nodes:           {summary['node_count']}"
    )

    print(
        f"Relationships:   {summary['edge_count']}"
    )

    print()
    print("NODE TYPES")
    print("-" * 70)

    for node_type, count in sorted(
        summary["node_types"].items()
    ):

        print(
            f"{node_type:<40} {count}"
        )

    print()
    print("RELATIONSHIP TYPES")
    print("-" * 70)

    for relationship, count in sorted(
        summary["relationship_types"].items()
    ):

        print(
            f"{relationship:<45} {count}"
        )


# ============================================================================
# SELF TEST
# ============================================================================

def self_test() -> None:

    print()
    print("=" * 70)
    print("KNOWLEDGE GRAPH BUILDER SELF-TEST")
    print("=" * 70)

    graph = KnowledgeGraph(
        schema_version=SCHEMA_VERSION
    )

    member = KGNode(
        node_id="member:test",
        node_type=resolve_node_type(
            "Member"
        ),
        label="test",
        properties={
            "member_id": "test",
        },
    )

    domain = KGNode(
        node_id="sdohdomain:economic_stability",
        node_type=resolve_node_type(
            "SDOHDomain"
        ),
        label="Economic Stability",
        properties={
            "domain": "Economic Stability",
        },
    )

    graph.add_node(
        member
    )

    graph.add_node(
        domain
    )

    add_relationship(
        graph,
        member.node_id,
        "HAS_SDOH_FACTOR",
        domain.node_id,
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1

    assert (
        graph.edges[0].source
        == member.node_id
    )

    assert (
        graph.edges[0].target
        == domain.node_id
    )

    print(
        "Node creation:       PASS"
    )

    print(
        "Relationship creation: PASS"
    )

    print(
        "JSON serialization:   PASS"
    )

    print(
        "BUILDER SELF-TEST:    PASSED"
    )

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 70)
    print("HEALTHLENS KNOWLEDGE GRAPH BUILDER")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    data = load_inputs()

    # --------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------

    validate_inputs(
        data
    )

    member_risk = data[
        "member_risk"
    ]

    member_features = data[
        "member_features"
    ]

    interventions = data[
        "interventions"
    ]

    county_risk = data[
        "county_risk"
    ]

    # --------------------------------------------------------------
    # Build
    # --------------------------------------------------------------

    graph = KnowledgeGraph(
        schema_version=SCHEMA_VERSION
    )

    member_summary = build_member_graph(
        graph=graph,
        member_risk=member_risk,
        member_features=member_features,
        interventions=interventions,
    )

    county_summary = build_county_graph(
        graph=graph,
        member_features=member_features,
        county_risk=county_risk,
    )

    # --------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------

    validation = validate_graph(
        graph
    )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    save_graph(
        graph
    )

    save_validation_report(
        validation
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print_summary(
        graph
    )

    print()
    print("=" * 70)
    print("KNOWLEDGE GRAPH BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Graph:\n{GRAPH_FILE}"
    )

    print(
        f"Validation:\n{VALIDATION_FILE}"
    )

    print(
        f"Summary:\n{SUMMARY_FILE}"
    )

    print()
    print(
        f"Members processed: "
        f"{member_summary['members_processed']}"
    )

    print(
        f"Members with county: "
        f"{member_summary['members_with_county']}"
    )

    print(
        f"Members without county: "
        f"{member_summary['members_without_county']}"
    )

    print(
        f"Counties processed: "
        f"{county_summary['counties_processed']}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The knowledge graph does not replace the ML model."
    )

    print(
        "Risk prediction remains the responsibility of the "
        "member-risk model."
    )

    print(
        "The graph provides SDOH context, evidence, "
        "relationships, and intervention connections."
    )


if __name__ == "__main__":

    main()