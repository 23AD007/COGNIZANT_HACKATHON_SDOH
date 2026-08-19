"""
HealthLens — Contextual Reasoning Engine

Purpose
-------
Combines:
    1. Member risk prediction
    2. Knowledge graph context
    3. Knowledge-base domains/factors
    4. Evidence
    5. Intervention mappings

The reasoning layer is deterministic.
It does NOT replace the ML risk model.

Architecture
------------
Risk Model
    |
    v
Knowledge Graph
    |
    v
Knowledge Base Registry
    |
    v
Contextual Reasoner
    |
    +--> SDOH drivers
    +--> Clinical context
    +--> Evidence
    +--> Intervention candidates
    |
    v
Explainable reasoning result
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
import json


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "knowledge_graph"
    / "healthlens_knowledge_graph.json"
)


# ============================================================================
# KNOWLEDGE BASE IMPORTS
# ============================================================================

from src.knowledge_base.registry import build_registry


# ============================================================================
# OPTIONAL NORMALIZATION HELPERS
# ============================================================================


def _normalize(value: Any) -> str:
    """
    Normalize identifiers for safe matching.

    Example:
        "Economic Stability"
        "economic_stability"
        "Economic-Stability"

    are normalized consistently.
    """

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def _canonical(value: Any) -> str:
    """
    Canonical string representation for factor/domain matching.
    """

    if value is None:
        return ""

    return str(value).strip()


# ============================================================================
# GENERIC OBJECT HELPERS
# ============================================================================


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """
    Safely retrieve a property from either:
        - dictionary
        - object/dataclass
    """

    if obj is None:
        return default

    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)

    return default


def _as_list(value: Any) -> list:
    """
    Normalize arbitrary iterable-like values to a list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    return [value]


# ============================================================================
# GRAPH LOADING
# ============================================================================


def load_graph(path: Path = GRAPH_PATH) -> dict:
    """
    Load the HealthLens knowledge graph JSON artifact.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge graph not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        graph = json.load(handle)

    if not isinstance(graph, Mapping):
        raise ValueError(
            "Knowledge graph must deserialize to a mapping."
        )

    return dict(graph)


# ============================================================================
# GRAPH STRUCTURE EXTRACTION
# ============================================================================


def _graph_nodes(graph: Mapping[str, Any]) -> list[dict]:
    """
    Extract graph nodes from supported graph artifact structures.
    """

    nodes = graph.get("nodes", [])

    if isinstance(nodes, Mapping):
        return [
            dict(node)
            for node in nodes.values()
            if isinstance(node, Mapping)
        ]

    if isinstance(nodes, list):
        return [
            dict(node)
            for node in nodes
            if isinstance(node, Mapping)
        ]

    return []


def _graph_relationships(graph: Mapping[str, Any]) -> list[dict]:
    """
    Extract graph relationships from supported graph artifact structures.
    """

    relationships = graph.get(
        "relationships",
        graph.get("edges", []),
    )

    if isinstance(relationships, Mapping):
        return [
            dict(rel)
            for rel in relationships.values()
            if isinstance(rel, Mapping)
        ]

    if isinstance(relationships, list):
        return [
            dict(rel)
            for rel in relationships
            if isinstance(rel, Mapping)
        ]

    return []


def _node_type(node: Mapping[str, Any]) -> str:
    return str(
        node.get(
            "type",
            node.get(
                "node_type",
                node.get("label", ""),
            ),
        )
    )


def _node_id(node: Mapping[str, Any]) -> str:
    return str(
        node.get(
            "id",
            node.get(
                "node_id",
                node.get("key", ""),
            ),
        )
    )


def _relationship_type(rel: Mapping[str, Any]) -> str:
    return str(
        rel.get(
            "type",
            rel.get(
                "relationship_type",
                rel.get("label", ""),
            ),
        )
    )


def _relationship_source(rel: Mapping[str, Any]) -> str:
    return str(
        rel.get(
            "source",
            rel.get(
                "from",
                rel.get(
                    "source_id",
                    rel.get("from_id", ""),
                ),
            ),
        )
    )


def _relationship_target(rel: Mapping[str, Any]) -> str:
    return str(
        rel.get(
            "target",
            rel.get(
                "to",
                rel.get(
                    "target_id",
                    rel.get("to_id", ""),
                ),
            ),
        )
    )


# ============================================================================
# MEMBER CONTEXT
# ============================================================================


@dataclass
class MemberContext:
    """
    Context extracted for one member.
    """

    member_id: str

    risk_probability: float | None = None

    risk_band: str | None = None

    sdoh_factors: list[str] = field(
        default_factory=list
    )

    sdoh_domains: list[str] = field(
        default_factory=list
    )

    clinical_factors: list[str] = field(
        default_factory=list
    )

    county: str | None = None

    county_fips: str | None = None

    county_name: str | None = None

    evidence_records: list[Any] = field(
        default_factory=list
    )

    intervention_candidates: list[Any] = field(
        default_factory=list
    )


# ============================================================================
# REASONING RESULT
# ============================================================================


@dataclass
class ReasoningResult:
    """
    Final explainable reasoning output.
    """

    member_id: str

    risk_probability: float | None = None

    risk_band: str | None = None

    sdoh_factors: list[str] = field(
        default_factory=list
    )

    sdoh_domains: list[str] = field(
        default_factory=list
    )

    clinical_factors: list[str] = field(
        default_factory=list
    )

    evidence_records: list[Any] = field(
        default_factory=list
    )

    intervention_candidates: list[Any] = field(
        default_factory=list
    )

    reasoning_trace: list[str] = field(
        default_factory=list
    )

    county: str | None = None

    county_fips: str | None = None

    county_name: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# CONTEXTUAL REASONER
# ============================================================================


class ContextualReasoner:
    """
    Deterministic contextual reasoning engine.

    The reasoner does not predict risk.

    It consumes risk-model output and enriches it with:
        - SDOH
        - clinical context
        - evidence
        - interventions
    """

    def __init__(
        self,
        graph: Mapping[str, Any] | None = None,
        registry: Any | None = None,
    ) -> None:

        self.graph = (
            dict(graph)
            if graph is not None
            else load_graph()
        )

        self.registry = (
            registry
            if registry is not None
            else build_registry()
        )

        self.nodes = _graph_nodes(
            self.graph
        )

        self.relationships = _graph_relationships(
            self.graph
        )

        self._nodes_by_id = {
            _node_id(node): node
            for node in self.nodes
            if _node_id(node)
        }

        self._member_nodes = [
            node
            for node in self.nodes
            if _node_type(node).upper() == "MEMBER"
        ]

        self._build_relationship_indexes()

    # ========================================================================
    # RELATIONSHIP INDEXES
    # ========================================================================

    def _build_relationship_indexes(self) -> None:

        self._outgoing: dict[str, list[dict]] = {}

        self._incoming: dict[str, list[dict]] = {}

        for rel in self.relationships:

            source = _relationship_source(rel)

            target = _relationship_target(rel)

            if source:
                self._outgoing.setdefault(
                    source,
                    [],
                ).append(rel)

            if target:
                self._incoming.setdefault(
                    target,
                    [],
                ).append(rel)

    # ========================================================================
    # MEMBER RESOLUTION
    # ========================================================================

    def resolve_member(
        self,
        member_id: str,
    ) -> dict | None:

        requested = str(
            member_id
        ).strip()

        if not requested:
            return None

        node = self._nodes_by_id.get(
            requested
        )

        if node is not None:
            if (
                _node_type(node).upper()
                == "MEMBER"
            ):
                return node

        for candidate in self._member_nodes:

            properties = candidate.get(
                "properties",
                {}
            )

            candidate_member_id = properties.get(
                "member_id"
            )

            if (
                candidate_member_id is not None
                and str(candidate_member_id).strip()
                == requested
            ):
                return candidate

        return None

    # ========================================================================
    # MEMBER RELATIONSHIPS
    # ========================================================================

    def _member_relationships(
        self,
        member_id: str,
        relationship_type: str | None = None,
    ) -> list[dict]:

        relationships = (
            self._outgoing.get(
                member_id,
                [],
            )
            + self._incoming.get(
                member_id,
                [],
            )
        )

        if relationship_type is None:
            return relationships

        wanted = relationship_type.upper()

        return [
            rel
            for rel in relationships
            if (
                _relationship_type(rel).upper()
                == wanted
            )
        ]

    # ========================================================================
    # NODE PROPERTY EXTRACTION
    # ========================================================================

    def _node_property(
        self,
        node: Mapping[str, Any],
        *names: str,
    ) -> Any:

        for name in names:

            if name in node:
                return node[name]

            properties = node.get(
                "properties"
            )

            if isinstance(
                properties,
                Mapping,
            ) and name in properties:
                return properties[name]

        return None

        # ========================================================================
    # RISK
    # ========================================================================

    def _extract_risk(
        self,
        member: Mapping[str, Any],
    ) -> tuple[
        float | None,
        str | None,
    ]:
        """
        Resolve the existing member risk assessment.

        IMPORTANT
        ---------
        The contextual reasoning layer does NOT calculate risk.

        Risk is produced upstream by the member-risk model and stored
        in the Knowledge Graph as a RiskAssessment node connected to
        the MEMBER through:

            MEMBER
                |
                | HAS_RISK_ASSESSMENT
                v
            RiskAssessment

        Resolution order
        ----------------
        1. Check the MEMBER node itself for backwards compatibility.
        2. Follow HAS_RISK_ASSESSMENT relationships.
        3. Inspect the connected RiskAssessment node.
        4. Inspect relationship properties as a final fallback.

        This keeps the reasoning layer deterministic and prevents it
        from creating or recalculating risk.
        """

        # ------------------------------------------------------------------
        # HELPERS
        # ------------------------------------------------------------------

        def extract_probability(
            record: Any,
        ) -> float | None:

            value = self._node_property(
                record,
                "risk_probability",
                "riskProbability",
                "probability",
                "risk_score",
                "riskScore",
                "score",
            )

            if value is None:
                return None

            try:
                value = float(value)
            except (
                TypeError,
                ValueError,
            ):
                return None

            # Risk probability must be a valid probability.
            if not 0.0 <= value <= 1.0:
                return None

            return value

        def extract_band(
            record: Any,
        ) -> str | None:

            value = self._node_property(
                record,
                "risk_band",
                "riskBand",
                "risk_category",
                "riskCategory",
                "band",
                "category",
            )

            if value is None:
                return None

            value = str(value).strip()

            if not value:
                return None

            return value

        # ------------------------------------------------------------------
        # 1. DIRECT MEMBER PROPERTIES
        # ------------------------------------------------------------------

        probability = extract_probability(
            member
        )

        band = extract_band(
            member
        )

        if (
            probability is not None
            or band is not None
        ):
            return (
                probability,
                band,
            )

        # ------------------------------------------------------------------
        # 2. RESOLVE MEMBER ID
        # ------------------------------------------------------------------

        member_id = _node_id(
            member
        )

        if not member_id:
            return (
                None,
                None,
            )

        # ------------------------------------------------------------------
        # 3. FOLLOW HAS_RISK_ASSESSMENT
        # ------------------------------------------------------------------

        risk_relationships = (
            self._member_relationships(
                member_id,
                "HAS_RISK_ASSESSMENT",
            )
        )

        # ------------------------------------------------------------------
        # 4. INSPECT RISK ASSESSMENT NODES
        # ------------------------------------------------------------------

        candidates: list[
            tuple[
                float | None,
                str | None,
                Any,
            ]
        ] = []

        for relationship in risk_relationships:

            source = _relationship_source(
                relationship
            )

            target = _relationship_target(
                relationship
            )

            # Determine the node on the opposite side
            # of the MEMBER.
            if source == member_id:
                risk_node_id = target

            elif target == member_id:
                risk_node_id = source

            else:
                continue

            if not risk_node_id:
                continue

            risk_node = (
                self._nodes_by_id.get(
                    risk_node_id
                )
            )

            if risk_node is None:
                continue

            # --------------------------------------------------------------
            # Make sure this is actually a risk-assessment node.
            # --------------------------------------------------------------

            node_type = _node_type(
                risk_node
            ).upper()

            if node_type not in {
                "RISKASSESSMENT",
                "RISK_ASSESSMENT",
            }:

                # Some graph serializers may omit the expected
                # node type, so do not immediately reject the node.
                #
                # Instead, accept it if it contains risk fields.
                has_risk_fields = any(
                    self._node_property(
                        risk_node,
                        field_name,
                    )
                    is not None
                    for field_name in (
                        "risk_probability",
                        "riskProbability",
                        "probability",
                        "risk_band",
                        "riskBand",
                        "risk_category",
                    )
                )

                if not has_risk_fields:
                    continue

            candidate_probability = (
                extract_probability(
                    risk_node
                )
            )

            candidate_band = (
                extract_band(
                    risk_node
                )
            )

            # --------------------------------------------------------------
            # Relationship fallback
            # --------------------------------------------------------------

            if candidate_probability is None:

                candidate_probability = (
                    extract_probability(
                        relationship
                    )
                )

            if candidate_band is None:

                candidate_band = (
                    extract_band(
                        relationship
                    )
                )

            if (
                candidate_probability is not None
                or candidate_band is not None
            ):

                candidates.append(
                    (
                        candidate_probability,
                        candidate_band,
                        risk_node,
                    )
                )

        # ------------------------------------------------------------------
        # 5. SELECT BEST RISK ASSESSMENT
        # ------------------------------------------------------------------

        if candidates:

            # Prefer candidates that contain a probability.
            #
            # If multiple assessments exist, preserve deterministic
            # behavior by:
            #
            #   1. preferring records with a probability
            #   2. preferring the latest timestamp/version when available
            #
            # We do NOT calculate a new probability.

            def candidate_key(
                candidate,
            ):

                candidate_probability = (
                    candidate[0]
                )

                candidate_node = (
                    candidate[2]
                )

                timestamp = (
                    self._node_property(
                        candidate_node,
                        "updated_at",
                        "updatedAt",
                        "created_at",
                        "createdAt",
                        "timestamp",
                        "date",
                    )
                )

                timestamp_string = (
                    str(timestamp)
                    if timestamp is not None
                    else ""
                )

                return (
                    candidate_probability
                    is not None,
                    timestamp_string,
                )

            candidates.sort(
                key=candidate_key,
                reverse=True,
            )

            probability, band, _ = (
                candidates[0]
            )

            return (
                probability,
                band,
            )

        # ------------------------------------------------------------------
        # 6. NO RISK ASSESSMENT FOUND
        # ------------------------------------------------------------------

        return (
            None,
            None,
        )
    # ========================================================================
    # SDOH FACTORS
    # ========================================================================

    def extract_sdoh_factors(
        self,
        member_id: str,
    ) -> list[str]:

        factors: list[str] = []

        for rel in self._member_relationships(
            member_id,
            "HAS_SDOH_FACTOR",
        ):

            source = _relationship_source(
                rel
            )

            target = _relationship_target(
                rel
            )

            other_id = (
                target
                if source == member_id
                else source
            )

            node = self._nodes_by_id.get(
                other_id
            )

            if node is None:
                continue

            factor = self._node_property(
                node,
                "factor",
                "factor_id",
                "name",
                "sdoh_factor",
            )

            if factor:
                factors.append(
                    str(factor)
                )

        # Deduplicate while preserving order.
        return list(
            dict.fromkeys(factors)
        )

    # ========================================================================
    # SDOH DOMAINS
    # ========================================================================

    def resolve_sdoh_domains(
        self,
        factors: Iterable[str],
    ) -> list[str]:

        domains: list[str] = []

        factor_set = {
            _normalize(factor)
            for factor in factors
        }

        # First use the registry as source of truth.
        registry_factors = self._registry_factors()

        for factor_id, factor_record in registry_factors.items():

            if (
                _normalize(factor_id)
                not in factor_set
            ):
                continue

            domain = _get(
                factor_record,
                "domain",
                "sdoh_domain",
                "domain_name",
            )

            if domain:
                domains.append(
                    str(domain)
                )

        # Preserve graph-derived domains if registry
        # metadata is not available.
        if not domains:

            for factor in factors:

                graph_domain = (
                    self._find_factor_domain(
                        factor
                    )
                )

                if graph_domain:
                    domains.append(
                        graph_domain
                    )

        return list(
            dict.fromkeys(domains)
        )

    # ========================================================================
    # GRAPH FACTOR DOMAIN
    # ========================================================================

    def _find_factor_domain(
        self,
        factor: str,
    ) -> str | None:

        wanted = _normalize(factor)

        for node in self.nodes:

            node_factor = self._node_property(
                node,
                "factor",
                "factor_id",
                "name",
            )

            if (
                _normalize(node_factor)
                != wanted
            ):
                continue

            for rel in self._outgoing.get(
                _node_id(node),
                [],
            ):

                if (
                    _relationship_type(rel).upper()
                    != "BELONGS_TO_DOMAIN"
                ):
                    continue

                target = _relationship_target(
                    rel
                )

                domain_node = (
                    self._nodes_by_id.get(
                        target
                    )
                )

                if domain_node:

                    domain = self._node_property(
                        domain_node,
                        "name",
                        "domain",
                        "domain_name",
                    )

                    if domain:
                        return str(domain)

        return None

    # ========================================================================
    # CLINICAL CONTEXT
    # ========================================================================

    def extract_clinical_factors(
        self,
        member_id: str,
    ) -> list[str]:

        factors: list[str] = []

        for rel in self._member_relationships(
            member_id,
            "HAS_CLINICAL_CONTEXT",
        ):

            source = _relationship_source(
                rel
            )

            target = _relationship_target(
                rel
            )

            other_id = (
                target
                if source == member_id
                else source
            )

            node = self._nodes_by_id.get(
                other_id
            )

            if node is None:
                continue

            name = self._node_property(
                node,
                "name",
                "factor",
                "clinical_factor",
                "feature",
            )

            if name:
                factors.append(
                    str(name)
                )

        return list(
            dict.fromkeys(factors)
        )

    # ========================================================================
    # COUNTY
    # ========================================================================

    def resolve_county(
        self,
        member_id: str,
    ) -> tuple[
        str | None,
        str | None,
        str | None,
    ]:

        relationships = (
            self._member_relationships(
                member_id,
                "LIVES_IN",
            )
        )

        for rel in relationships:

            source = _relationship_source(
                rel
            )

            target = _relationship_target(
                rel
            )

            county_id = (
                target
                if source == member_id
                else source
            )

            county = self._nodes_by_id.get(
                county_id
            )

            if county is None:
                continue

            fips = self._node_property(
                county,
                "fips",
                "county_fips",
                "FIPS",
            )

            name = self._node_property(
                county,
                "name",
                "county_name",
            )

            return (
                county_id,
                str(fips)
                if fips is not None
                else None,
                str(name)
                if name is not None
                else None,
            )

        return (
            None,
            None,
            None,
        )

    # ========================================================================
    # REGISTRY FACTORS
    # ========================================================================

    def _registry_factors(self) -> dict[str, Any]:
        """
        Return registry SDOH factors indexed by canonical factor ID.

        Supports several registry implementations without changing
        the registry itself.
        """

        registry = self.registry

        # Preferred public APIs.
        for attribute in (
            "factors",
            "sdoh_factors",
            "factor_registry",
        ):

            value = getattr(
                registry,
                attribute,
                None,
            )

            if isinstance(
                value,
                Mapping,
            ):

                result = {}

                for key, record in value.items():

                    factor_id = (
                        _get(
                            record,
                            "factor",
                            "factor_id",
                            "id",
                            "name",
                            default=key,
                        )
                    )

                    if factor_id:
                        result[
                            str(factor_id)
                        ] = record

                if result:
                    return result

        # Public lookup fallback.
        getter = getattr(
            registry,
            "get_factor",
            None,
        )

        if callable(getter):

            result = {}

            # Registry may expose a collection.
            collection = getattr(
                registry,
                "factor_ids",
                None,
            )

            if collection:

                for factor_id in collection:

                    record = getter(
                        factor_id
                    )

                    if record is not None:
                        result[
                            str(factor_id)
                        ] = record

            if result:
                return result

        return {}

    # ========================================================================
    # REGISTRY INTERVENTIONS
    # ========================================================================

    def _registry_interventions(
        self,
    ) -> dict[str, Any]:

        registry = self.registry

        for attribute in (
            "interventions",
            "intervention_registry",
        ):

            value = getattr(
                registry,
                attribute,
                None,
            )

            if isinstance(
                value,
                Mapping,
            ):

                result = {}

                for key, record in value.items():

                    intervention_id = _get(
                        record,
                        "intervention_id",
                        "id",
                        "key",
                        default=key,
                    )

                    if intervention_id:
                        result[
                            str(intervention_id)
                        ] = record

                if result:
                    return result

        getter = getattr(
            registry,
            "get_intervention",
            None,
        )

        collection = getattr(
            registry,
            "intervention_ids",
            None,
        )

        if callable(getter) and collection:

            result = {}

            for intervention_id in collection:

                record = getter(
                    intervention_id
                )

                if record is not None:
                    result[
                        str(intervention_id)
                    ] = record

            if result:
                return result

        return {}

    # ========================================================================
    # INTERVENTION TARGET FACTORS
    # ========================================================================

    def _intervention_target_factors(
        self,
        intervention: Any,
    ) -> list[str]:

        targets = _get(
            intervention,
            "target_factors",
            "target_factor_ids",
            "factors",
            "sdoh_factors",
            "targetFactors",
            default=[],
        )

        result: list[str] = []

        for target in _as_list(targets):

            if isinstance(
                target,
                Mapping,
            ):

                value = _get(
                    target,
                    "factor",
                    "factor_id",
                    "id",
                    "name",
                )

            else:

                value = target

            if value:
                result.append(
                    str(value)
                )

        return list(
            dict.fromkeys(result)
        )

    # ========================================================================
    # INTERVENTION MATCHING
    # ========================================================================

    def resolve_intervention_candidates(
        self,
        factors: Iterable[str],
        domains: Iterable[str] | None = None,
    ) -> list[dict]:

        """
        Resolve interventions using canonical registry factors.

        Matching rules
        --------------
        1. Exact canonical factor match.
        2. Normalized factor match.
        3. Domain match is used only as a fallback.

        Candidates are ranked by:
            - number of matched factors
            - domain match
            - intervention ID

        This prevents the reasoner from inventing interventions.
        """

        member_factors = list(
            dict.fromkeys(
                str(factor)
                for factor in factors
                if factor
            )
        )

        member_factor_normalized = {
            _normalize(factor)
            for factor in member_factors
        }

        member_domains = list(
            domains or []
        )

        member_domain_normalized = {
            _normalize(domain)
            for domain in member_domains
        }

        interventions = (
            self._registry_interventions()
        )

        candidates: list[dict] = []

        for intervention_id, intervention in interventions.items():

            target_factors = (
                self._intervention_target_factors(
                    intervention
                )
            )

            matched_factors = []

            for target_factor in target_factors:

                if (
                    _normalize(target_factor)
                    in member_factor_normalized
                ):

                    matched_factors.append(
                        target_factor
                    )

            intervention_domain = _get(
                intervention,
                "domain",
                "sdoh_domain",
                "domain_name",
            )

            domain_match = False

            if intervention_domain:

                domain_match = (
                    _normalize(
                        intervention_domain
                    )
                    in member_domain_normalized
                )

            # Factor match is the primary criterion.
            if matched_factors:

                candidates.append(
                    {
                        "intervention_id":
                            str(intervention_id),

                        "name": _get(
                            intervention,
                            "name",
                            "title",
                            "label",
                            default=str(
                                intervention_id
                            ),
                        ),

                        "domain":
                            intervention_domain,

                        "matched_factors":
                            list(
                                dict.fromkeys(
                                    matched_factors
                                )
                            ),

                        "matched_factor_count":
                            len(
                                matched_factors
                            ),

                        "domain_match":
                            domain_match,

                        "score":
                            (
                                len(
                                    matched_factors
                                )
                                * 10
                                + (
                                    1
                                    if domain_match
                                    else 0
                                )
                            ),
                    }
                )

        # Domain-only fallback.
        #
        # This is intentionally secondary. It allows an intervention
        # such as a broad social-support intervention to remain eligible
        # even when it has no explicit factor targets.
        if not candidates:

            for intervention_id, intervention in interventions.items():

                intervention_domain = _get(
                    intervention,
                    "domain",
                    "sdoh_domain",
                    "domain_name",
                )

                if not intervention_domain:
                    continue

                if (
                    _normalize(
                        intervention_domain
                    )
                    not in member_domain_normalized
                ):
                    continue

                candidates.append(
                    {
                        "intervention_id":
                            str(intervention_id),

                        "name": _get(
                            intervention,
                            "name",
                            "title",
                            "label",
                            default=str(
                                intervention_id
                            ),
                        ),

                        "domain":
                            intervention_domain,

                        "matched_factors": [],

                        "matched_factor_count":
                            0,

                        "domain_match":
                            True,

                        "score":
                            1,
                    }
                )

        candidates.sort(
            key=lambda item: (
                -int(
                    item.get(
                        "score",
                        0,
                    )
                ),
                -int(
                    item.get(
                        "matched_factor_count",
                        0,
                    )
                ),
                str(
                    item.get(
                        "intervention_id",
                        "",
                    )
                ),
            )
        )

        return candidates

    # ========================================================================
    # EVIDENCE
    # ========================================================================

    def resolve_evidence(
        self,
        factors: Iterable[str],
    ) -> list[Any]:

        factor_set = {
            _normalize(factor)
            for factor in factors
        }

        evidence: list[Any] = []

        # Prefer registry APIs.
        getter = getattr(
            self.registry,
            "get_evidence_for_factor",
            None,
        )

        if callable(getter):

            for factor in factors:

                try:

                    records = getter(
                        factor
                    )

                except Exception:
                    records = []

                for record in _as_list(
                    records
                ):

                    evidence.append(
                        record
                    )

        # Registry collection fallback.
        if not evidence:

            records = getattr(
                self.registry,
                "evidence",
                None,
            )

            if records is None:
                records = getattr(
                    self.registry,
                    "evidence_records",
                    None,
                )

            for record in _as_list(
                records
            ):

                factor = _get(
                    record,
                    "factor",
                )

                if (
                    _normalize(factor)
                    in factor_set
                ):

                    evidence.append(
                        record
                    )

        # Deduplicate.
        unique: list[Any] = []

        seen: set[str] = set()

        for record in evidence:

            evidence_id = _get(
                record,
                "evidence_id",
                "id",
            )

            key = (
                str(evidence_id)
                if evidence_id
                else repr(record)
            )

            if key in seen:
                continue

            seen.add(key)

            unique.append(record)

        return unique

    # ========================================================================
    # MEMBER CONTEXT
    # ========================================================================

    def build_member_context(
        self,
        member_id: str,
    ) -> MemberContext:

        member = self.resolve_member(
            member_id
        )

        if member is None:

            raise ValueError(
                f"Member not found in graph: "
                f"{member_id}"
            )

        resolved_id = _node_id(
            member
        )

        probability, band = (
            self._extract_risk(
                member
            )
        )

        factors = (
            self.extract_sdoh_factors(
                resolved_id
            )
        )

        domains = (
            self.resolve_sdoh_domains(
                factors
            )
        )

        clinical = (
            self.extract_clinical_factors(
                resolved_id
            )
        )

        county_id, county_fips, county_name = (
            self.resolve_county(
                resolved_id
            )
        )

        evidence = (
            self.resolve_evidence(
                factors
            )
        )

        interventions = (
            self.resolve_intervention_candidates(
                factors,
                domains,
            )
        )

        return MemberContext(
            member_id=resolved_id,
            risk_probability=probability,
            risk_band=band,
            sdoh_factors=factors,
            sdoh_domains=domains,
            clinical_factors=clinical,
            county=county_id,
            county_fips=county_fips,
            county_name=county_name,
            evidence_records=evidence,
            intervention_candidates=interventions,
        )

    # ========================================================================
    # REASONING
    # ========================================================================

    def reason(
        self,
        member_id: str,
    ) -> ReasoningResult:

        trace: list[str] = []

        trace.append(
            "Member resolved."
        )

        context = (
            self.build_member_context(
                member_id
            )
        )

        trace.append(
            f"Resolved {len(context.sdoh_factors)} "
            f"SDOH factors."
        )

        trace.append(
            f"Resolved {len(context.sdoh_domains)} "
            f"SDOH domains."
        )

        trace.append(
            f"Resolved {len(context.clinical_factors)} "
            f"clinical factors."
        )

        trace.append(
            f"Resolved {len(context.evidence_records)} "
            f"evidence records."
        )

        trace.append(
            f"Resolved "
            f"{len(context.intervention_candidates)} "
            f"intervention candidates."
        )

        return ReasoningResult(
            member_id=context.member_id,
            risk_probability=context.risk_probability,
            risk_band=context.risk_band,
            sdoh_factors=context.sdoh_factors,
            sdoh_domains=context.sdoh_domains,
            clinical_factors=context.clinical_factors,
            evidence_records=context.evidence_records,
            intervention_candidates=(
                context.intervention_candidates
            ),
            reasoning_trace=trace,
            county=context.county,
            county_fips=context.county_fips,
            county_name=context.county_name,
        )


# ============================================================================
# SERIALIZATION
# ============================================================================


def _serialize_object(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _serialize_object(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _serialize_object(item)
            for item in value
        ]

    if hasattr(
        value,
        "__dataclass_fields__",
    ):
        return _serialize_object(
            asdict(value)
        )

    if hasattr(
        value,
        "__dict__",
    ):
        return _serialize_object(
            vars(value)
        )

    return str(value)


def serialize_reasoning_result(
    result: ReasoningResult,
) -> dict:

    return _serialize_object(
        result.to_dict()
    )


# ============================================================================
# SELF TEST
# ============================================================================


def _self_test() -> None:

    print("=" * 78)
    print(
        "HEALTHLENS CONTEXTUAL REASONING ENGINE"
    )
    print("=" * 78)

    # ------------------------------------------------------------------------
    # GRAPH
    # ------------------------------------------------------------------------

    graph = load_graph()

    nodes = _graph_nodes(
        graph
    )

    relationships = _graph_relationships(
        graph
    )

    print(
        "Knowledge Graph loading:       PASS"
    )

    print(
        f"Graph nodes detected:          "
        f"{len(nodes)}"
    )

    print(
        f"Graph relationships detected:  "
        f"{len(relationships)}"
    )

    # ------------------------------------------------------------------------
    # REGISTRY
    # ------------------------------------------------------------------------

    registry = build_registry()

    print(
        "Knowledge Base loading:        PASS"
    )

    # ------------------------------------------------------------------------
    # REASONER
    # ------------------------------------------------------------------------

    reasoner = ContextualReasoner(
        graph=graph,
        registry=registry,
    )

    # Use the known test member from the current
    # HealthLens graph when available.
    member_id = None

    if reasoner._member_nodes:

        first_member = (
            reasoner._member_nodes[0]
        )

        member_id = _node_id(
            first_member
        )

    if not member_id:

        raise AssertionError(
            "No MEMBER node found in knowledge graph."
        )

    print(
        f"Test member:                    "
        f"{member_id}"
    )

    print(
        "Reasoning engine construction: PASS"
    )

    # ------------------------------------------------------------------------
    # MEMBER RESOLUTION
    # ------------------------------------------------------------------------

    member = reasoner.resolve_member(
        member_id
    )

    assert member is not None

    print(
        "Member resolution:             PASS"
    )

    # ------------------------------------------------------------------------
    # CONTEXT
    # ------------------------------------------------------------------------

    context = reasoner.build_member_context(
        member_id
    )

    assert context.member_id == member_id

    print(
        "Member context extraction:      PASS"
    )

    # ------------------------------------------------------------------------
    # REASONING
    # ------------------------------------------------------------------------

    result = reasoner.reason(
        member_id
    )

    assert result.member_id == member_id

    print(
        "Member reasoning:               PASS"
    )

    # ------------------------------------------------------------------------
    # FACTORS
    # ------------------------------------------------------------------------

    assert isinstance(
        result.sdoh_factors,
        list,
    )

    print(
        "Member ID resolution:            PASS"
    )

    print(
        "SDOH factor extraction:          PASS"
    )

    # ------------------------------------------------------------------------
    # DOMAINS
    # ------------------------------------------------------------------------

    assert isinstance(
        result.sdoh_domains,
        list,
    )

    print(
        "SDOH domain resolution:          PASS"
    )

    # ------------------------------------------------------------------------
    # CLINICAL
    # ------------------------------------------------------------------------

    assert isinstance(
        result.clinical_factors,
        list,
    )

    print(
        "Clinical context extraction:     PASS"
    )

    # ------------------------------------------------------------------------
    # EVIDENCE
    # ------------------------------------------------------------------------

    assert isinstance(
        result.evidence_records,
        list,
    )

    print(
        "Evidence resolution:             PASS"
    )

    # ------------------------------------------------------------------------
    # INTERVENTIONS
    # ------------------------------------------------------------------------

    assert isinstance(
        result.intervention_candidates,
        list,
    )

    if result.intervention_candidates:

        print(
            "Intervention candidate lookup:   PASS"
        )

    else:

        print(
            "Intervention candidate lookup:   PASS (none detected)"
        )

    # ------------------------------------------------------------------------
    # SERIALIZATION
    # ------------------------------------------------------------------------

    serialized = (
        serialize_reasoning_result(
            result
        )
    )

    assert isinstance(
        serialized,
        dict,
    )

    assert (
        serialized["member_id"]
        == member_id
    )

    print(
        "Reasoning serialization:         PASS"
    )

    # ------------------------------------------------------------------------
    # TRACE
    # ------------------------------------------------------------------------

    assert result.reasoning_trace

    print(
        "Reasoning trace:                 PASS"
    )

    # ------------------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "REASONING RESULT"
    )
    print("=" * 78)

    print(
        f"Member ID:              "
        f"{result.member_id}"
    )

    print(
        f"Risk probability:       "
        f"{result.risk_probability}"
    )

    print(
        f"Risk band:              "
        f"{result.risk_band}"
    )

    print(
        f"SDOH factors:           "
        f"{len(result.sdoh_factors)}"
    )

    print(
        f"SDOH domains:           "
        f"{len(result.sdoh_domains)}"
    )

    print(
        f"Clinical factors:       "
        f"{len(result.clinical_factors)}"
    )

    print(
        f"Evidence records:       "
        f"{len(result.evidence_records)}"
    )

    print(
        f"Intervention candidates:"
        f" {len(result.intervention_candidates)}"
    )

    print()
    print("DOMAINS")

    for domain in result.sdoh_domains:

        print(
            f"  - {domain}"
        )

    print()
    print(
        "RELEVANT SDOH FACTORS"
    )

    for factor in result.sdoh_factors:

        print(
            f"  - {factor}"
        )

    print()
    print(
        "INTERVENTION CANDIDATES"
    )

    for candidate in (
        result.intervention_candidates
    ):

        print(
            f"  - "
            f"{candidate.get('intervention_id')}"
            f" | "
            f"{candidate.get('name')}"
            f" | matched factors: "
            f"{candidate.get('matched_factor_count')}"
            f" | score: "
            f"{candidate.get('score')}"
        )

    print()
    print("=" * 78)
    print(
        "CONTEXTUAL REASONING SELF-TEST: PASSED"
    )
    print("=" * 78)


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "MemberContext",
    "ReasoningResult",
    "ContextualReasoner",
    "load_graph",
    "serialize_reasoning_result",
]


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    _self_test()
