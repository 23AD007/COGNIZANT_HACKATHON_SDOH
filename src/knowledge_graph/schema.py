"""
Knowledge Graph schema for HealthLens SDOH risk prioritization.

This module contains:
- Node types
- Evidence types
- Relationship types
- Canonical node/edge dataclasses
- Schema validation
- Serialization helpers
- Self-test

The schema is intentionally independent of pandas, sklearn, Neo4j,
NetworkX, or any other graph library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional


# ======================================================================
# ENUMS
# ======================================================================


class NodeType(str, Enum):
    """
    Canonical node categories in the HealthLens knowledge graph.
    """

    MEMBER = "Member"
    PATIENT = "Patient"
    COUNTY = "County"
    SDOH_FACTOR = "SDOHFactor"
    SDOH_DOMAIN = "SDOHDomain"
    CLINICAL_CONDITION = "ClinicalCondition"
    CLINICAL_ENCOUNTER = "ClinicalEncounter"
    INTERVENTION = "Intervention"
    RISK_SCORE = "RiskScore"


class EvidenceType(str, Enum):
    """
    Source/evidence categories used to explain graph relationships.
    """

    MEMBER_RISK = "member_risk"
    COUNTY_RISK = "county_risk"
    SDOH_FEATURE = "sdoh_feature"
    CLINICAL_FEATURE = "clinical_feature"
    TARGET_OUTCOME = "target_outcome"
    INTERVENTION_PRIORITY = "intervention_priority"
    DATA_SOURCE = "data_source"
    DERIVED = "derived"


class RelationshipType(str, Enum):
    """
    Canonical relationships between graph nodes.
    """

    HAS_RISK = "HAS_RISK"
    LIVES_IN = "LIVES_IN"
    HAS_SDOH_FACTOR = "HAS_SDOH_FACTOR"
    BELONGS_TO_DOMAIN = "BELONGS_TO_DOMAIN"
    HAS_CLINICAL_CONDITION = "HAS_CLINICAL_CONDITION"
    HAS_ENCOUNTER = "HAS_ENCOUNTER"
    HAS_INTERVENTION_PRIORITY = "HAS_INTERVENTION_PRIORITY"
    RECOMMENDS = "RECOMMENDS"
    AFFECTS = "AFFECTS"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    SUPPORTS = "SUPPORTS"
    DERIVED_FROM = "DERIVED_FROM"


# ======================================================================
# CONSTANTS
# ======================================================================


SCHEMA_VERSION = "1.0.0"


VALID_NODE_TYPES = {item.value for item in NodeType}
VALID_EVIDENCE_TYPES = {item.value for item in EvidenceType}
VALID_RELATIONSHIP_TYPES = {item.value for item in RelationshipType}


# ======================================================================
# NORMALIZATION HELPERS
# ======================================================================


def _clean_string(value: Any) -> Optional[str]:
    """
    Convert a value to a clean string.

    Empty strings and None become None.
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def _clean_properties(
    properties: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Remove None keys/values from arbitrary property mappings.
    """
    if properties is None:
        return {}

    cleaned: Dict[str, Any] = {}

    for key, value in properties.items():
        if key is None:
            continue

        key_text = str(key).strip()

        if not key_text:
            continue

        if value is None:
            continue

        cleaned[key_text] = value

    return cleaned


# ======================================================================
# NODE
# ======================================================================


@dataclass
class KGNode:
    """
    Canonical knowledge graph node.
    """

    node_id: str
    node_type: str
    label: str

    properties: Dict[str, Any] = field(default_factory=dict)

    source: Optional[str] = None
    evidence_type: Optional[str] = None

    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        self.node_id = _clean_string(self.node_id) or ""
        self.label = _clean_string(self.label) or ""

        self.node_type = normalize_node_type(self.node_type)

        self.properties = _clean_properties(self.properties)

        if self.source is not None:
            self.source = _clean_string(self.source)

        if self.evidence_type is not None:
            self.evidence_type = normalize_evidence_type(
                self.evidence_type
            )

        if self.confidence is not None:
            self.confidence = float(self.confidence)

            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "confidence must be between 0 and 1."
                )

        self.validate()

    def validate(self) -> None:
        if not self.node_id:
            raise ValueError("node_id cannot be empty.")

        if not self.label:
            raise ValueError("label cannot be empty.")

        if self.node_type not in VALID_NODE_TYPES:
            raise ValueError(
                f"Invalid node_type: {self.node_type}"
            )

        if (
            self.evidence_type is not None
            and self.evidence_type not in VALID_EVIDENCE_TYPES
        ):
            raise ValueError(
                f"Invalid evidence_type: {self.evidence_type}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize node into a graph-friendly dictionary.
        """

        data = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "properties": dict(self.properties),
        }

        if self.source is not None:
            data["source"] = self.source

        if self.evidence_type is not None:
            data["evidence_type"] = self.evidence_type

        if self.confidence is not None:
            data["confidence"] = self.confidence

        return data


# ======================================================================
# EDGE
# ======================================================================


@dataclass
class KGEdge:
    """
    Directed relationship between two graph nodes.
    """

    source_id: str
    relationship_type: str
    target_id: str

    properties: Dict[str, Any] = field(default_factory=dict)

    evidence_type: Optional[str] = None
    source: Optional[str] = None

    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        self.source_id = _clean_string(self.source_id) or ""
        self.target_id = _clean_string(self.target_id) or ""

        self.relationship_type = normalize_relationship_type(
            self.relationship_type
        )

        self.properties = _clean_properties(self.properties)

        if self.evidence_type is not None:
            self.evidence_type = normalize_evidence_type(
                self.evidence_type
            )

        if self.source is not None:
            self.source = _clean_string(self.source)

        if self.confidence is not None:
            self.confidence = float(self.confidence)

            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "confidence must be between 0 and 1."
                )

        self.validate()

    def validate(self) -> None:
        if not self.source_id:
            raise ValueError("source_id cannot be empty.")

        if not self.target_id:
            raise ValueError("target_id cannot be empty.")

        if not self.relationship_type:
            raise ValueError(
                "relationship_type cannot be empty."
            )

        if self.relationship_type not in VALID_RELATIONSHIP_TYPES:
            raise ValueError(
                "Invalid relationship_type: "
                f"{self.relationship_type}"
            )

        if (
            self.evidence_type is not None
            and self.evidence_type not in VALID_EVIDENCE_TYPES
        ):
            raise ValueError(
                f"Invalid evidence_type: {self.evidence_type}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize edge into a graph-friendly dictionary.
        """

        data = {
            "source_id": self.source_id,
            "relationship_type": self.relationship_type,
            "target_id": self.target_id,
            "properties": dict(self.properties),
        }

        if self.evidence_type is not None:
            data["evidence_type"] = self.evidence_type

        if self.source is not None:
            data["source"] = self.source

        if self.confidence is not None:
            data["confidence"] = self.confidence

        return data


# ======================================================================
# GRAPH SCHEMA
# ======================================================================


@dataclass
class GraphSchema:
    """
    Complete schema description.
    """

    version: str = SCHEMA_VERSION

    node_types: List[str] = field(
        default_factory=lambda: sorted(VALID_NODE_TYPES)
    )

    evidence_types: List[str] = field(
        default_factory=lambda: sorted(VALID_EVIDENCE_TYPES)
    )

    relationship_types: List[str] = field(
        default_factory=lambda: sorted(VALID_RELATIONSHIP_TYPES)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "node_types": list(self.node_types),
            "evidence_types": list(self.evidence_types),
            "relationship_types": list(self.relationship_types),
        }


# ======================================================================
# NORMALIZATION
# ======================================================================


def normalize_node_type(value: Any) -> str:
    """
    Normalize NodeType enum or string to canonical value.
    """

    if isinstance(value, NodeType):
        return value.value

    text = _clean_string(value)

    if text is None:
        raise ValueError("node_type cannot be empty.")

    aliases = {
        "member": NodeType.MEMBER.value,
        "patient": NodeType.PATIENT.value,
        "county": NodeType.COUNTY.value,
        "sdoh": NodeType.SDOH_FACTOR.value,
        "sdoh_factor": NodeType.SDOH_FACTOR.value,
        "sdohfactor": NodeType.SDOH_FACTOR.value,
        "sdoh_domain": NodeType.SDOH_DOMAIN.value,
        "sdohdomain": NodeType.SDOH_DOMAIN.value,
        "condition": NodeType.CLINICAL_CONDITION.value,
        "clinical_condition": NodeType.CLINICAL_CONDITION.value,
        "encounter": NodeType.CLINICAL_ENCOUNTER.value,
        "clinical_encounter": NodeType.CLINICAL_ENCOUNTER.value,
        "intervention": NodeType.INTERVENTION.value,
        "risk": NodeType.RISK_SCORE.value,
        "risk_score": NodeType.RISK_SCORE.value,
    }

    return aliases.get(text.lower(), text)


def normalize_evidence_type(value: Any) -> str:
    """
    Normalize EvidenceType enum or string.
    """

    if isinstance(value, EvidenceType):
        return value.value

    text = _clean_string(value)

    if text is None:
        raise ValueError("evidence_type cannot be empty.")

    aliases = {
        "member risk": EvidenceType.MEMBER_RISK.value,
        "member_risk": EvidenceType.MEMBER_RISK.value,
        "county risk": EvidenceType.COUNTY_RISK.value,
        "county_risk": EvidenceType.COUNTY_RISK.value,
        "sdoh": EvidenceType.SDOH_FEATURE.value,
        "sdoh feature": EvidenceType.SDOH_FEATURE.value,
        "sdoh_feature": EvidenceType.SDOH_FEATURE.value,
        "clinical": EvidenceType.CLINICAL_FEATURE.value,
        "clinical feature": EvidenceType.CLINICAL_FEATURE.value,
        "clinical_feature": EvidenceType.CLINICAL_FEATURE.value,
        "target": EvidenceType.TARGET_OUTCOME.value,
        "target outcome": EvidenceType.TARGET_OUTCOME.value,
        "target_outcome": EvidenceType.TARGET_OUTCOME.value,
        "intervention": EvidenceType.INTERVENTION_PRIORITY.value,
        "intervention priority": (
            EvidenceType.INTERVENTION_PRIORITY.value
        ),
        "intervention_priority": (
            EvidenceType.INTERVENTION_PRIORITY.value
        ),
        "derived": EvidenceType.DERIVED.value,
    }

    return aliases.get(text.lower(), text)


def normalize_relationship_type(value: Any) -> str:
    """
    Normalize RelationshipType enum or string.
    """

    if isinstance(value, RelationshipType):
        return value.value

    text = _clean_string(value)

    if text is None:
        raise ValueError(
            "relationship_type cannot be empty."
        )

    aliases = {
        "has risk": RelationshipType.HAS_RISK.value,
        "has_risk": RelationshipType.HAS_RISK.value,

        "lives in": RelationshipType.LIVES_IN.value,
        "lives_in": RelationshipType.LIVES_IN.value,

        "has sdoh factor": (
            RelationshipType.HAS_SDOH_FACTOR.value
        ),
        "has_sdoh_factor": (
            RelationshipType.HAS_SDOH_FACTOR.value
        ),

        "belongs to domain": (
            RelationshipType.BELONGS_TO_DOMAIN.value
        ),
        "belongs_to_domain": (
            RelationshipType.BELONGS_TO_DOMAIN.value
        ),

        "has clinical condition": (
            RelationshipType.HAS_CLINICAL_CONDITION.value
        ),
        "has_clinical_condition": (
            RelationshipType.HAS_CLINICAL_CONDITION.value
        ),

        "has encounter": (
            RelationshipType.HAS_ENCOUNTER.value
        ),
        "has_encounter": (
            RelationshipType.HAS_ENCOUNTER.value
        ),

        "has intervention priority": (
            RelationshipType.HAS_INTERVENTION_PRIORITY.value
        ),
        "has_intervention_priority": (
            RelationshipType.HAS_INTERVENTION_PRIORITY.value
        ),

        "recommends": RelationshipType.RECOMMENDS.value,
        "affects": RelationshipType.AFFECTS.value,
        "associated with": (
            RelationshipType.ASSOCIATED_WITH.value
        ),
        "associated_with": (
            RelationshipType.ASSOCIATED_WITH.value
        ),
        "supports": RelationshipType.SUPPORTS.value,
        "derived from": (
            RelationshipType.DERIVED_FROM.value
        ),
        "derived_from": (
            RelationshipType.DERIVED_FROM.value
        ),
    }

    normalized = text.strip()

    return aliases.get(
        normalized.lower(),
        normalized.upper(),
    )


# ======================================================================
# FACTORY HELPERS
# ======================================================================


def make_member_node(
    member_id: str,
    *,
    patient_id: Optional[str] = None,
    county_fips: Optional[Any] = None,
    risk_probability: Optional[float] = None,
    risk_band: Optional[str] = None,
) -> KGNode:
    """
    Create a canonical Member node.
    """

    properties: Dict[str, Any] = {}

    if patient_id is not None:
        properties["patient_id"] = patient_id

    if county_fips is not None:
        properties["county_fips"] = county_fips

    if risk_probability is not None:
        properties["risk_probability"] = float(
            risk_probability
        )

    if risk_band is not None:
        properties["risk_band"] = risk_band

    return KGNode(
        node_id=f"member:{member_id}",
        node_type=NodeType.MEMBER.value,
        label=str(member_id),
        properties=properties,
        source="member_risk_scores.csv",
        evidence_type=EvidenceType.MEMBER_RISK.value,
    )


def make_county_node(
    county_fips: Any,
    *,
    county_name: Optional[str] = None,
    state_fips: Optional[Any] = None,
    risk_probability: Optional[float] = None,
    risk_band: Optional[str] = None,
) -> KGNode:
    """
    Create a canonical County node.
    """

    county_text = str(county_fips)

    properties: Dict[str, Any] = {
        "county_fips": county_fips,
    }

    if county_name is not None:
        properties["county_name"] = county_name

    if state_fips is not None:
        properties["state_fips"] = state_fips

    if risk_probability is not None:
        properties["risk_probability"] = float(
            risk_probability
        )

    if risk_band is not None:
        properties["risk_band"] = risk_band

    return KGNode(
        node_id=f"county:{county_text}",
        node_type=NodeType.COUNTY.value,
        label=county_text,
        properties=properties,
        source="county_risk_scores.csv",
        evidence_type=EvidenceType.COUNTY_RISK.value,
    )


def make_sdoh_factor_node(
    factor_name: str,
    *,
    domain: Optional[str] = None,
    value: Optional[Any] = None,
    percentile: Optional[Any] = None,
) -> KGNode:
    """
    Create an SDOH factor node.
    """

    properties: Dict[str, Any] = {}

    if domain is not None:
        properties["domain"] = domain

    if value is not None:
        properties["value"] = value

    if percentile is not None:
        properties["percentile"] = percentile

    return KGNode(
        node_id=f"sdoh_factor:{factor_name}",
        node_type=NodeType.SDOH_FACTOR.value,
        label=factor_name,
        properties=properties,
        source="member_model_features.csv",
        evidence_type=EvidenceType.SDOH_FEATURE.value,
    )


def make_domain_node(domain: str) -> KGNode:
    """
    Create an SDOH domain node.
    """

    return KGNode(
        node_id=f"sdoh_domain:{domain}",
        node_type=NodeType.SDOH_DOMAIN.value,
        label=domain,
        properties={},
        source="intervention_priorities.csv",
        evidence_type=EvidenceType.INTERVENTION_PRIORITY.value,
    )


def make_intervention_node(
    intervention_name: str,
    *,
    domain: Optional[str] = None,
) -> KGNode:
    """
    Create an intervention node.
    """

    properties: Dict[str, Any] = {}

    if domain is not None:
        properties["domain"] = domain

    return KGNode(
        node_id=f"intervention:{intervention_name}",
        node_type=NodeType.INTERVENTION.value,
        label=intervention_name,
        properties=properties,
        source="intervention_priorities.csv",
        evidence_type=EvidenceType.INTERVENTION_PRIORITY.value,
    )


# ======================================================================
# EDGE FACTORIES
# ======================================================================


def make_edge(
    source_id: str,
    relationship_type: Any,
    target_id: str,
    *,
    properties: Optional[Mapping[str, Any]] = None,
    evidence_type: Optional[Any] = None,
    source: Optional[str] = None,
    confidence: Optional[float] = None,
) -> KGEdge:
    """
    Generic edge factory.
    """

    return KGEdge(
        source_id=source_id,
        relationship_type=normalize_relationship_type(
            relationship_type
        ),
        target_id=target_id,
        properties=dict(properties or {}),
        evidence_type=(
            normalize_evidence_type(evidence_type)
            if evidence_type is not None
            else None
        ),
        source=source,
        confidence=confidence,
    )


# ======================================================================
# GRAPH VALIDATION
# ======================================================================


def validate_graph(
    nodes: Iterable[KGNode],
    edges: Iterable[KGEdge],
) -> Dict[str, Any]:
    """
    Validate a collection of nodes and edges.

    Returns a validation summary rather than raising for ordinary
    graph-level problems.
    """

    node_list = list(nodes)
    edge_list = list(edges)

    node_ids = [node.node_id for node in node_list]

    duplicate_nodes = sorted(
        {
            node_id
            for node_id in node_ids
            if node_ids.count(node_id) > 1
        }
    )

    node_id_set = set(node_ids)

    missing_edge_sources = sorted(
        {
            edge.source_id
            for edge in edge_list
            if edge.source_id not in node_id_set
        }
    )

    missing_edge_targets = sorted(
        {
            edge.target_id
            for edge in edge_list
            if edge.target_id not in node_id_set
        }
    )

    errors: List[str] = []

    if duplicate_nodes:
        errors.append(
            f"Duplicate node IDs: {duplicate_nodes}"
        )

    if missing_edge_sources:
        errors.append(
            "Missing edge source nodes: "
            f"{missing_edge_sources}"
        )

    if missing_edge_targets:
        errors.append(
            "Missing edge target nodes: "
            f"{missing_edge_targets}"
        )

    return {
        "valid": len(errors) == 0,
        "node_count": len(node_list),
        "edge_count": len(edge_list),
        "duplicate_nodes": duplicate_nodes,
        "missing_edge_sources": missing_edge_sources,
        "missing_edge_targets": missing_edge_targets,
        "errors": errors,
    }


# ======================================================================
# SERIALIZATION
# ======================================================================


def schema_to_dict() -> Dict[str, Any]:
    """
    Return complete schema metadata.
    """

    return GraphSchema().to_dict()


def node_to_dict(node: KGNode) -> Dict[str, Any]:
    return node.to_dict()


def edge_to_dict(edge: KGEdge) -> Dict[str, Any]:
    return edge.to_dict()


# ======================================================================
# SELF TEST
# ======================================================================


def _run_schema_self_test() -> None:
    """
    Internal schema sanity test.

    Run with:

        py -3.12 -m src.knowledge_graph.schema
    """

    assert NodeType.MEMBER.value == "Member"
    assert NodeType.COUNTY.value == "County"
    assert NodeType.SDOH_FACTOR.value == "SDOHFactor"

    assert (
        EvidenceType.MEMBER_RISK.value
        == "member_risk"
    )

    assert (
        RelationshipType.HAS_RISK.value
        == "HAS_RISK"
    )

    assert (
        normalize_node_type("member")
        == "Member"
    )

    assert (
        normalize_evidence_type("member risk")
        == "member_risk"
    )

    assert (
        normalize_relationship_type("has risk")
        == "HAS_RISK"
    )

    member = make_member_node(
        "test-member",
        patient_id="test-patient",
        county_fips=25017,
        risk_probability=0.82,
        risk_band="High",
    )

    county = make_county_node(
        25017,
        county_name="Example County",
        risk_probability=0.55,
        risk_band="High",
    )

    edge = make_edge(
        member.node_id,
        RelationshipType.LIVES_IN,
        county.node_id,
        evidence_type=EvidenceType.SDOH_FEATURE,
    )

    validation = validate_graph(
        [member, county],
        [edge],
    )

    assert validation["valid"] is True
    assert validation["node_count"] == 2
    assert validation["edge_count"] == 1

    assert member.to_dict()["node_type"] == "Member"
    assert county.to_dict()["node_type"] == "County"
    assert edge.to_dict()["relationship_type"] == "LIVES_IN"

    print("=" * 70)
    print("KNOWLEDGE GRAPH SCHEMA SELF-TEST")
    print("=" * 70)
    print(f"Schema version:       {SCHEMA_VERSION}")
    print(f"Node types:           {len(VALID_NODE_TYPES)}")
    print(f"Evidence types:       {len(VALID_EVIDENCE_TYPES)}")
    print(f"Relationship types:   {len(VALID_RELATIONSHIP_TYPES)}")
    print()
    print("NodeType:             PASS")
    print("EvidenceType:         PASS")
    print("RelationshipType:     PASS")
    print("Node factory:         PASS")
    print("Edge factory:         PASS")
    print("Graph validation:     PASS")
    print()
    print("SCHEMA SELF-TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    _run_schema_self_test()