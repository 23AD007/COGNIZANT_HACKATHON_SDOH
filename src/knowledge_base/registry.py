"""
HealthLens Knowledge Base Registry
==================================

Single read-only integration layer for:

    src.knowledge_base.sdoh_domains
    src.knowledge_base.sdoh_factors
    src.knowledge_base.interventions
    src.knowledge_base.evidence

Responsibilities
----------------
This registry:

    1. Loads the authoritative Knowledge Base modules.
    2. Normalizes their different object representations.
    3. Resolves SDOH domains consistently.
    4. Creates stable identifiers for evidence records.
    5. Builds cross-reference indexes.
    6. Validates Knowledge Base consistency.
    7. Provides read-only lookup APIs.

This module does NOT modify:

    - ML risk scores
    - member risk generation
    - county risk generation
    - intervention prioritization
    - knowledge graph data

The Knowledge Base remains the source of truth.
The registry is only an integration layer.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping
import importlib
import re


# ============================================================================
# VERSION
# ============================================================================

REGISTRY_VERSION = "1.0.2"
SCHEMA_VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# MODULE IMPORTS
# ============================================================================

def _load_module(module_name: str):
    """
    Import a Knowledge Base module.
    """

    try:
        return importlib.import_module(module_name)

    except ModuleNotFoundError as exc:

        raise ImportError(
            f"Unable to import Knowledge Base module: "
            f"{module_name}\n"
            f"Run this command from the HealthLens project root:\n"
            f"    py -3.12 -m src.knowledge_base.registry"
        ) from exc


DOMAINS_MODULE = _load_module(
    "src.knowledge_base.sdoh_domains"
)

FACTORS_MODULE = _load_module(
    "src.knowledge_base.sdoh_factors"
)

INTERVENTIONS_MODULE = _load_module(
    "src.knowledge_base.interventions"
)

EVIDENCE_MODULE = _load_module(
    "src.knowledge_base.evidence"
)


# ============================================================================
# GENERIC NORMALIZATION
# ============================================================================

def _normalize(value: Any) -> str:
    """
    Normalize a value for ordinary identifier matching.

    Examples:

        Economic Stability
        economic_stability

    both become:

        economic_stability
    """

    if value is None:
        return ""

    value = str(value).strip().lower()

    value = re.sub(
        r"[\s\-]+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value


def _compact_normalize(value: Any) -> str:
    """
    Normalize a value aggressively for semantic matching.

    Examples:

        Economic Stability
        EconomicStability
        economic_stability

    all become:

        economicstability
    """

    if value is None:
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).strip().lower(),
    )


def _clean_string(
    value: Any,
) -> str | None:
    """
    Return a stripped string or None.
    """

    if value is None:
        return None

    value = str(value).strip()

    return value if value else None


def _to_dict(
    value: Any,
) -> dict[str, Any]:
    """
    Convert a Knowledge Base object to a plain dictionary.

    Supported:

        Mapping
        dataclass
        Pydantic-like model_dump()
        ordinary objects exposing __dict__
    """

    if isinstance(value, Mapping):
        return dict(value)

    if is_dataclass(value) and not isinstance(
        value,
        type,
    ):
        return asdict(value)

    if hasattr(value, "model_dump"):

        dumped = value.model_dump()

        if isinstance(dumped, Mapping):
            return dict(dumped)

    if hasattr(value, "__dict__"):

        return {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_")
        }

    raise TypeError(
        "Unsupported Knowledge Base object type: "
        f"{type(value).__name__}"
    )


# ============================================================================
# DOMAIN NORMALIZATION
# ============================================================================

def _domain_id_from_record(
    record: Mapping[str, Any],
) -> str | None:

    value = (
        record.get("id")
        or record.get("domain_id")
        or record.get("code")
        or record.get("slug")
        or record.get("key")
    )

    if value:
        return str(value)

    return None


def _domain_name_from_record(
    record: Mapping[str, Any],
) -> str | None:

    value = (
        record.get("name")
        or record.get("display_name")
        or record.get("label")
        or record.get("title")
        or record.get("domain_name")
    )

    if value:
        return str(value)

    return None


# ============================================================================
# FACTOR NORMALIZATION
# ============================================================================

def _factor_id_from_record(
    record: Mapping[str, Any],
) -> str | None:

    value = (
        record.get("id")
        or record.get("factor_id")
        or record.get("code")
        or record.get("slug")
        or record.get("key")
        or record.get("name")
        or record.get("factor_name")
    )

    if value:
        return str(value)

    return None


# ============================================================================
# INTERVENTION NORMALIZATION
# ============================================================================

def _intervention_id_from_record(
    record: Mapping[str, Any],
) -> str | None:

    value = (
        record.get("id")
        or record.get("intervention_id")
        or record.get("code")
        or record.get("slug")
        or record.get("key")
        or record.get("name")
        or record.get("intervention_name")
    )

    if value:
        return str(value)

    return None


# ============================================================================
# EVIDENCE NORMALIZATION
# ============================================================================

def _evidence_id_from_record(
    record: Mapping[str, Any],
) -> str | None:
    """
    Resolve a stable evidence ID.

    EvidenceRecord currently has no explicit evidence_id.

    Therefore the factor is used as the stable identity because
    each evidence record corresponds to an SDOH factor.

    Example:

        factor =
            snap_households_count_sum

    becomes:

        evidence_snap_households_count_sum
    """

    value = (
        record.get("id")
        or record.get("evidence_id")
        or record.get("code")
        or record.get("slug")
        or record.get("key")
        or record.get("name")
        or record.get("title")
    )

    if value:
        return str(value)

    factor = (
        record.get("factor")
        or record.get("factor_id")
        or record.get("sdoh_factor")
        or record.get("sdoh_factor_id")
    )

    if factor:

        normalized_factor = _normalize(
            factor
        )

        return (
            "evidence_"
            f"{normalized_factor}"
        )

    return None


# ============================================================================
# EVIDENCE DOMAIN NORMALIZATION
# ============================================================================

EVIDENCE_DOMAIN_ALIASES = {
    "economicstability":
        "economic_stability",

    "educationaccess":
        "education_access",

    "healthcareaccess":
        "healthcare_access",

    "neighborhoodbuiltenvironment":
        "neighborhood_built_environment",

    "socialcommunitycontext":
        "social_community_context",

    "housing":
        "housing",

    "transportation":
        "transportation",

    "digitalaccess":
        "digital_access",
}


def _normalize_evidence_domain(
    value: Any,
) -> Any:
    """
    Convert compact EvidenceRecord domain names into
    canonical registry-compatible domain identifiers.
    """

    if value is None:
        return value

    compact = _compact_normalize(
        value
    )

    return EVIDENCE_DOMAIN_ALIASES.get(
        compact,
        _normalize(value),
    )


# ============================================================================
# EXTRACT MODULE DATA
# ============================================================================

def _extract_from_known_functions(
    module: Any,
    function_names: tuple[str, ...],
) -> list[Any]:
    """
    Extract records from a known public Knowledge Base function.

    Supported return types:

        Mapping
        list
        tuple
        set
        single object
        None
    """

    for function_name in function_names:

        function = getattr(
            module,
            function_name,
            None,
        )

        if not callable(function):
            continue

        try:
            result = function()

        except TypeError:
            continue

        if result is None:
            continue

        if isinstance(
            result,
            Mapping,
        ):

            return list(
                result.values()
            )

        if isinstance(
            result,
            (list, tuple, set),
        ):

            return list(result)

        return [result]

    return []


def _extract_module_constants(
    module: Any,
) -> list[Any]:
    """
    Fallback extraction from public module constants.

    This is intentionally conservative.
    """

    candidates: list[Any] = []

    for name, value in vars(module).items():

        if name.startswith("_"):
            continue

        if isinstance(
            value,
            Mapping,
        ):

            candidates.extend(
                value.values()
            )

        elif isinstance(
            value,
            (list, tuple, set),
        ):

            candidates.extend(
                value
            )

    return candidates


def _extract_records(
    module: Any,
    function_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    """
    Extract and normalize Knowledge Base records.
    """

    raw_records = _extract_from_known_functions(
        module,
        function_names,
    )

    if not raw_records:

        raw_records = _extract_module_constants(
            module
        )

    records: list[dict[str, Any]] = []

    for item in raw_records:

        try:

            record = _to_dict(
                item
            )

            records.append(
                record
            )

        except TypeError:

            continue

    return records


# ============================================================================
# REGISTRY
# ============================================================================

class KnowledgeBaseRegistry:

    def __init__(self) -> None:

        self.registry_version = (
            REGISTRY_VERSION
        )

        self.schema_version = (
            SCHEMA_VERSION
        )

        # --------------------------------------------------------------
        # PRIMARY REGISTRIES
        # --------------------------------------------------------------

        self.domains: dict[
            str,
            dict[str, Any],
        ] = {}

        self.domain_names: dict[
            str,
            str,
        ] = {}

        self.domain_compact_names: dict[
            str,
            str,
        ] = {}

        self.factors: dict[
            str,
            dict[str, Any],
        ] = {}

        self.interventions: dict[
            str,
            dict[str, Any],
        ] = {}

        self.evidence: dict[
            str,
            dict[str, Any],
        ] = {}

        # --------------------------------------------------------------
        # INDEXES
        # --------------------------------------------------------------

        self.factor_by_feature: dict[
            str,
            str,
        ] = {}

        self.factor_by_domain: dict[
            str,
            list[str],
        ] = {}

        self.interventions_by_factor: dict[
            str,
            list[str],
        ] = {}

        self.interventions_by_domain: dict[
            str,
            list[str],
        ] = {}

        self.evidence_by_factor: dict[
            str,
            list[str],
        ] = {}

        self.evidence_by_intervention: dict[
            str,
            list[str],
        ] = {}

        # --------------------------------------------------------------
        # LOAD + INDEX + VALIDATE
        # --------------------------------------------------------------

        self._load()

        self._build_indexes()

        self.validation_errors = (
            self.validate()
        )

    # ========================================================================
    # LOAD
    # ========================================================================

    def _load(self) -> None:

        # ----------------------------------------------------------------
        # DOMAINS
        # ----------------------------------------------------------------

        domain_records = _extract_records(
            DOMAINS_MODULE,
            (
                "get_all_sdoh_domains",
                "get_all_domains",
                "get_sdoh_domains",
            ),
        )

        for record in domain_records:

            domain_id = (
                _domain_id_from_record(
                    record
                )
            )

            domain_name = (
                _domain_name_from_record(
                    record
                )
            )

            if not domain_id:

                if domain_name:

                    domain_id = (
                        _normalize(
                            domain_name
                        )
                    )

            if not domain_id:
                continue

            key = _normalize(
                domain_id
            )

            record["_registry_id"] = (
                domain_id
            )

            record["_registry_name"] = (
                domain_name
                or domain_id
            )

            self.domains[key] = (
                record
            )

            self.domain_names[
                _normalize(
                    domain_name
                    or domain_id
                )
            ] = key

            self.domain_compact_names[
                _compact_normalize(
                    domain_name
                    or domain_id
                )
            ] = key

            self.domain_compact_names[
                _compact_normalize(
                    domain_id
                )
            ] = key

        # ----------------------------------------------------------------
        # FACTORS
        # ----------------------------------------------------------------

        factor_records = _extract_records(
            FACTORS_MODULE,
            (
                "get_all_sdoh_factors",
                "get_all_factors",
                "get_sdoh_factors",
            ),
        )

        for record in factor_records:

            factor_id = (
                _factor_id_from_record(
                    record
                )
            )

            if not factor_id:
                continue

            key = _normalize(
                factor_id
            )

            record["_registry_id"] = (
                factor_id
            )

            self.factors[key] = (
                record
            )

        # ----------------------------------------------------------------
        # INTERVENTIONS
        # ----------------------------------------------------------------

        intervention_records = _extract_records(
            INTERVENTIONS_MODULE,
            (
                "get_all_interventions",
                "get_interventions",
            ),
        )

        for record in intervention_records:

            intervention_id = (
                _intervention_id_from_record(
                    record
                )
            )

            if not intervention_id:
                continue

            key = _normalize(
                intervention_id
            )

            record["_registry_id"] = (
                intervention_id
            )

            self.interventions[key] = (
                record
            )

        # ----------------------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------------------

        evidence_records = _extract_records(
            EVIDENCE_MODULE,
            (
                "get_all_evidence",
                "get_evidence",
                "get_all_evidence_records",
                "get_evidence_records",
                "list_evidence",
            ),
        )

        for record in evidence_records:

            # Normalize domain representation.
            if "domain" in record:

                record["domain"] = (
                    _normalize_evidence_domain(
                        record["domain"]
                    )
                )

            evidence_id = (
                _evidence_id_from_record(
                    record
                )
            )

            if not evidence_id:
                continue

            key = _normalize(
                evidence_id
            )

            record["_registry_id"] = (
                evidence_id
            )

            self.evidence[key] = (
                record
            )

    # ========================================================================
    # DOMAIN RESOLUTION
    # ========================================================================

    def resolve_domain(
        self,
        value: Any,
    ) -> str | None:
        """
        Resolve a domain from:

            economic_stability
            Economic Stability
            EconomicStability
            economic-stability
        """

        if value is None:
            return None

        normalized = _normalize(
            value
        )

        # Direct registry key.
        if normalized in self.domains:
            return normalized

        # Human-readable name.
        if normalized in self.domain_names:

            return self.domain_names[
                normalized
            ]

        # Compact comparison.
        compact = _compact_normalize(
            value
        )

        if compact in self.domain_compact_names:

            return self.domain_compact_names[
                compact
            ]

        # Final comparison.
        for domain_id, record in (
            self.domains.items()
        ):

            domain_name = (
                record.get(
                    "_registry_name"
                )
            )

            if not domain_name:
                continue

            if (
                _compact_normalize(
                    domain_name
                )
                == compact
            ):

                return domain_id

        return None

    # ========================================================================
    # REFERENCE EXTRACTION
    # ========================================================================

    @staticmethod
    def _references(
        record: Mapping[str, Any],
        keys: tuple[str, ...],
    ) -> list[str]:
        """
        Extract one or more references from a record.
        """

        values: list[str] = []

        for key in keys:

            if key not in record:
                continue

            value = record[key]

            if value is None:
                continue

            if isinstance(
                value,
                str,
            ):

                if value.strip():

                    values.append(
                        value.strip()
                    )

            elif isinstance(
                value,
                Mapping,
            ):

                for item in value.values():

                    if item is not None:

                        values.append(
                            str(item)
                        )

            elif isinstance(
                value,
                (list, tuple, set),
            ):

                for item in value:

                    if item is not None:

                        values.append(
                            str(item)
                        )

            else:

                values.append(
                    str(value)
                )

        # Deduplicate while preserving order.
        result: list[str] = []

        seen: set[str] = set()

        for value in values:

            key = _normalize(
                value
            )

            if key and key not in seen:

                seen.add(key)

                result.append(
                    value
                )

        return result

    # ========================================================================
    # INDEXES
    # ========================================================================

    def _build_indexes(self) -> None:

        # ----------------------------------------------------------------
        # FACTORS
        # ----------------------------------------------------------------

        for factor_id, factor in (
            self.factors.items()
        ):

            # Feature → factor.
            #
            # Current SDOHFactor objects use `key` as the model feature.
            feature = (
                factor.get("model_feature")
                or factor.get("feature")
                or factor.get("feature_name")
                or factor.get("source_feature")
                or factor.get("key")
            )

            if feature:

                self.factor_by_feature[
                    _normalize(feature)
                ] = factor_id

            # Factor → domain.
            domain_values = (
                self._references(
                    factor,
                    (
                        "domain",
                        "domain_id",
                        "domain_key",
                        "sdoh_domain",
                        "sdoh_domain_id",
                    ),
                )
            )

            for domain_value in domain_values:

                domain_id = (
                    self.resolve_domain(
                        domain_value
                    )
                )

                if domain_id is None:
                    continue

                values = (
                    self.factor_by_domain.setdefault(
                        domain_id,
                        [],
                    )
                )

                if factor_id not in values:

                    values.append(
                        factor_id
                    )

        # ----------------------------------------------------------------
        # INTERVENTIONS
        # ----------------------------------------------------------------

        for (
            intervention_id,
            intervention,
        ) in self.interventions.items():

            # Intervention → domain.
            domain_values = (
                self._references(
                    intervention,
                    (
                        "domain",
                        "domain_id",
                        "sdoh_domain",
                        "sdoh_domain_id",
                        "sdoh_domains",
                    ),
                )
            )

            for domain_value in domain_values:

                domain_id = (
                    self.resolve_domain(
                        domain_value
                    )
                )

                if domain_id is None:
                    continue

                values = (
                    self.interventions_by_domain.setdefault(
                        domain_id,
                        [],
                    )
                )

                if (
                    intervention_id
                    not in values
                ):

                    values.append(
                        intervention_id
                    )

            # Intervention → factor.
            factor_values = (
                self._references(
                    intervention,
                    (
                        "sdoh_factor",
                        "sdoh_factor_id",
                        "factor",
                        "factor_id",
                        "sdoh_factors",
                        "factor_ids",
                        "target_factors",
                    ),
                )
            )

            for factor_value in factor_values:

                factor_id = _normalize(
                    factor_value
                )

                if factor_id not in self.factors:
                    continue

                values = (
                    self.interventions_by_factor.setdefault(
                        factor_id,
                        [],
                    )
                )

                if (
                    intervention_id
                    not in values
                ):

                    values.append(
                        intervention_id
                    )

        # ----------------------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------------------

        for (
            evidence_id,
            evidence,
        ) in self.evidence.items():

            factor_values = (
                self._references(
                    evidence,
                    (
                        "sdoh_factor",
                        "sdoh_factor_id",
                        "factor",
                        "factor_id",
                        "sdoh_factors",
                        "factor_ids",
                    ),
                )
            )

            for factor_value in factor_values:

                factor_id = _normalize(
                    factor_value
                )

                if factor_id not in self.factors:
                    continue

                values = (
                    self.evidence_by_factor.setdefault(
                        factor_id,
                        [],
                    )
                )

                if evidence_id not in values:

                    values.append(
                        evidence_id
                    )

                # EvidenceRecord currently does not store a concrete
                # intervention ID. Its intervention_link field is a
                # boolean semantic indicator.
                #
                # Therefore we connect evidence to interventions through
                # the factor → intervention relationship.
                intervention_ids = (
                    self.interventions_by_factor.get(
                        factor_id,
                        [],
                    )
                )

                for intervention_id in (
                    intervention_ids
                ):

                    intervention_values = (
                        self.evidence_by_intervention.setdefault(
                            intervention_id,
                            [],
                        )
                    )

                    if (
                        evidence_id
                        not in intervention_values
                    ):

                        intervention_values.append(
                            evidence_id
                        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate(
        self,
    ) -> list[str]:
        """
        Validate Knowledge Base consistency.
        """

        errors: list[str] = []

        # ----------------------------------------------------------------
        # EMPTY REGISTRIES
        # ----------------------------------------------------------------

        if not self.domains:

            errors.append(
                "No SDOH domains were loaded."
            )

        if not self.factors:

            errors.append(
                "No SDOH factors were loaded."
            )

        if not self.interventions:

            errors.append(
                "No interventions were loaded."
            )

        # Evidence is expected for this project.
        if not self.evidence:

            errors.append(
                "No evidence records were loaded."
            )

        # ----------------------------------------------------------------
        # DOMAIN VALIDATION
        # ----------------------------------------------------------------

        for (
            domain_id,
            domain,
        ) in self.domains.items():

            if not domain_id:

                errors.append(
                    "A domain has an empty registry ID."
                )

            name = (
                domain.get(
                    "_registry_name"
                )
            )

            if not name:

                errors.append(
                    f"Domain '{domain_id}' "
                    "has no name."
                )

        # ----------------------------------------------------------------
        # FACTOR VALIDATION
        # ----------------------------------------------------------------

        for (
            factor_id,
            factor,
        ) in self.factors.items():

            if not factor_id:

                errors.append(
                    "A factor has an empty registry ID."
                )

            domain_values = (
                self._references(
                    factor,
                    (
                        "domain",
                        "domain_id",
                        "domain_key",
                        "sdoh_domain",
                        "sdoh_domain_id",
                    ),
                )
            )

            if not domain_values:

                errors.append(
                    f"Factor '{factor_id}' "
                    "has no SDOH domain."
                )

            for domain_value in (
                domain_values
            ):

                if (
                    self.resolve_domain(
                        domain_value
                    )
                    is None
                ):

                    errors.append(
                        f"Factor "
                        f"'{factor_id}' references "
                        f"unknown SDOH domain "
                        f"'{domain_value}'."
                    )

        # ----------------------------------------------------------------
        # INTERVENTION VALIDATION
        # ----------------------------------------------------------------

        for (
            intervention_id,
            intervention,
        ) in self.interventions.items():

            domain_values = (
                self._references(
                    intervention,
                    (
                        "domain",
                        "domain_id",
                        "sdoh_domain",
                        "sdoh_domain_id",
                        "sdoh_domains",
                    ),
                )
            )

            for domain_value in (
                domain_values
            ):

                if (
                    self.resolve_domain(
                        domain_value
                    )
                    is None
                ):

                    errors.append(
                        f"Intervention "
                        f"'{intervention_id}' "
                        f"references unknown "
                        f"SDOH domain "
                        f"'{domain_value}'."
                    )

            factor_values = (
                self._references(
                    intervention,
                    (
                        "sdoh_factor",
                        "sdoh_factor_id",
                        "factor",
                        "factor_id",
                        "sdoh_factors",
                        "factor_ids",
                        "target_factors",
                    ),
                )
            )

            for factor_value in (
                factor_values
            ):

                if (
                    _normalize(
                        factor_value
                    )
                    not in self.factors
                ):

                    errors.append(
                        f"Intervention "
                        f"'{intervention_id}' "
                        f"references unknown "
                        f"SDOH factor "
                        f"'{factor_value}'."
                    )

        # ----------------------------------------------------------------
        # EVIDENCE VALIDATION
        # ----------------------------------------------------------------

        for (
            evidence_id,
            evidence,
        ) in self.evidence.items():

            factor_values = (
                self._references(
                    evidence,
                    (
                        "sdoh_factor",
                        "sdoh_factor_id",
                        "factor",
                        "factor_id",
                        "sdoh_factors",
                        "factor_ids",
                    ),
                )
            )

            if not factor_values:

                errors.append(
                    f"Evidence "
                    f"'{evidence_id}' has no "
                    "SDOH factor reference."
                )

            for factor_value in (
                factor_values
            ):

                if (
                    _normalize(
                        factor_value
                    )
                    not in self.factors
                ):

                    errors.append(
                        f"Evidence "
                        f"'{evidence_id}' references "
                        f"unknown SDOH factor "
                        f"'{factor_value}'."
                    )

            # Evidence domain validation.
            domain_value = evidence.get(
                "domain"
            )

            if domain_value is not None:

                if (
                    self.resolve_domain(
                        domain_value
                    )
                    is None
                ):

                    errors.append(
                        f"Evidence "
                        f"'{evidence_id}' references "
                        f"unknown SDOH domain "
                        f"'{domain_value}'."
                    )

        # ----------------------------------------------------------------
        # EVIDENCE DUPLICATE CHECK
        # ----------------------------------------------------------------

        evidence_ids = list(
            self.evidence.keys()
        )

        if len(evidence_ids) != len(
            set(evidence_ids)
        ):

            errors.append(
                "Duplicate evidence IDs detected."
            )

        return errors

    def assert_valid(self) -> None:
        """
        Raise if registry validation fails.
        """

        if self.validation_errors:

            details = "\n".join(
                f"  - {error}"
                for error in self.validation_errors
            )

            raise ValueError(
                "Knowledge Base registry validation failed:\n"
                f"{details}"
            )

    # ========================================================================
    # LOOKUPS
    # ========================================================================

    def get_domain(
        self,
        domain: str,
    ) -> dict[str, Any] | None:

        domain_id = (
            self.resolve_domain(
                domain
            )
        )

        if domain_id is None:
            return None

        return self.domains.get(
            domain_id
        )

    def get_factor(
        self,
        factor_id: str,
    ) -> dict[str, Any] | None:

        return self.factors.get(
            _normalize(
                factor_id
            )
        )

    def get_factor_by_feature(
        self,
        feature_name: str,
    ) -> dict[str, Any] | None:

        factor_id = (
            self.factor_by_feature.get(
                _normalize(
                    feature_name
                )
            )
        )

        if factor_id is None:
            return None

        return self.get_factor(
            factor_id
        )

    def get_intervention(
        self,
        intervention_id: str,
    ) -> dict[str, Any] | None:

        return self.interventions.get(
            _normalize(
                intervention_id
            )
        )

    def get_evidence(
        self,
        evidence_id: str,
    ) -> dict[str, Any] | None:

        return self.evidence.get(
            _normalize(
                evidence_id
            )
        )

    def get_factors_for_domain(
        self,
        domain: str,
    ) -> list[dict[str, Any]]:

        domain_id = (
            self.resolve_domain(
                domain
            )
        )

        if domain_id is None:
            return []

        factor_ids = (
            self.factor_by_domain.get(
                domain_id,
                [],
            )
        )

        return [
            self.factors[factor_id]
            for factor_id in factor_ids
            if factor_id in self.factors
        ]

    def get_interventions_for_domain(
        self,
        domain: str,
    ) -> list[dict[str, Any]]:

        domain_id = (
            self.resolve_domain(
                domain
            )
        )

        if domain_id is None:
            return []

        intervention_ids = (
            self.interventions_by_domain.get(
                domain_id,
                [],
            )
        )

        return [
            self.interventions[
                intervention_id
            ]
            for intervention_id
            in intervention_ids
            if intervention_id
            in self.interventions
        ]

    def get_interventions_for_factor(
        self,
        factor_id: str,
    ) -> list[dict[str, Any]]:

        intervention_ids = (
            self.interventions_by_factor.get(
                _normalize(
                    factor_id
                ),
                [],
            )
        )

        return [
            self.interventions[
                intervention_id
            ]
            for intervention_id
            in intervention_ids
            if intervention_id
            in self.interventions
        ]

    def get_evidence_for_factor(
        self,
        factor_id: str,
    ) -> list[dict[str, Any]]:

        evidence_ids = (
            self.evidence_by_factor.get(
                _normalize(
                    factor_id
                ),
                [],
            )
        )

        return [
            self.evidence[
                evidence_id
            ]
            for evidence_id
            in evidence_ids
            if evidence_id
            in self.evidence
        ]

    def get_evidence_for_intervention(
        self,
        intervention_id: str,
    ) -> list[dict[str, Any]]:

        evidence_ids = (
            self.evidence_by_intervention.get(
                _normalize(
                    intervention_id
                ),
                [],
            )
        )

        return [
            self.evidence[
                evidence_id
            ]
            for evidence_id
            in evidence_ids
            if evidence_id
            in self.evidence
        ]

    # ========================================================================
    # CONTEXT
    # ========================================================================

    def get_factor_context(
        self,
        factor_id: str,
    ) -> dict[str, Any]:

        factor = self.get_factor(
            factor_id
        )

        if factor is None:

            raise KeyError(
                f"Unknown SDOH factor: "
                f"{factor_id}"
            )

        domain_values = (
            self._references(
                factor,
                (
                    "domain",
                    "domain_id",
                    "domain_key",
                    "sdoh_domain",
                    "sdoh_domain_id",
                ),
            )
        )

        domains = []

        for domain in domain_values:

            resolved = (
                self.get_domain(
                    domain
                )
            )

            if resolved is not None:

                domains.append(
                    resolved
                )

        interventions = (
            self.get_interventions_for_factor(
                factor_id
            )
        )

        evidence = (
            self.get_evidence_for_factor(
                factor_id
            )
        )

        return {
            "factor": factor,
            "domains": domains,
            "interventions": interventions,
            "evidence": evidence,
        }

    # ========================================================================
    # SUMMARY
    # ========================================================================

    def summary(
        self,
    ) -> dict[str, Any]:

        return {

            "registry_version":
                self.registry_version,

            "schema_version":
                self.schema_version,

            "domains":
                len(self.domains),

            "factors":
                len(self.factors),

            "interventions":
                len(self.interventions),

            "evidence":
                len(self.evidence),

            "factor_feature_mappings":
                len(self.factor_by_feature),

            "factor_domain_mappings":
                sum(
                    len(values)
                    for values
                    in self.factor_by_domain.values()
                ),

            "intervention_factor_mappings":
                sum(
                    len(values)
                    for values
                    in self.interventions_by_factor.values()
                ),

            "intervention_domain_mappings":
                sum(
                    len(values)
                    for values
                    in self.interventions_by_domain.values()
                ),

            "evidence_factor_mappings":
                sum(
                    len(values)
                    for values
                    in self.evidence_by_factor.values()
                ),

            "evidence_intervention_mappings":
                sum(
                    len(values)
                    for values
                    in self.evidence_by_intervention.values()
                ),

            "validation_passed":
                not bool(
                    self.validation_errors
                ),

            "validation_error_count":
                len(
                    self.validation_errors
                ),
        }


# ============================================================================
# FACTORY
# ============================================================================

def build_registry() -> KnowledgeBaseRegistry:
    """
    Build and validate the Knowledge Base registry.
    """

    registry = (
        KnowledgeBaseRegistry()
    )

    registry.assert_valid()

    return registry


# ============================================================================
# SELF TEST
# ============================================================================

def _run_registry_self_test() -> None:

    print("=" * 70)
    print(
        "HEALTHLENS KNOWLEDGE BASE REGISTRY"
    )
    print("=" * 70)

    registry = build_registry()

    summary = registry.summary()

    print()
    print("REGISTRY")
    print("-" * 70)

    print(
        f"Registry version:     "
        f"{REGISTRY_VERSION}"
    )

    print(
        f"Schema version:       "
        f"{SCHEMA_VERSION}"
    )

    print()
    print("KNOWLEDGE BASE COUNTS")
    print("-" * 70)

    print(
        f"SDOH domains:          "
        f"{summary['domains']}"
    )

    print(
        f"SDOH factors:          "
        f"{summary['factors']}"
    )

    print(
        f"Interventions:         "
        f"{summary['interventions']}"
    )

    print(
        f"Evidence records:      "
        f"{summary['evidence']}"
    )

    print()
    print("VALIDATION")
    print("-" * 70)

    print(
        "Registry validation:   PASS"
        if summary["validation_passed"]
        else "Registry validation:   FAIL"
    )

    # ----------------------------------------------------------------
    # DOMAIN LOOKUP
    # ----------------------------------------------------------------

    domain_test = (
        "Economic Stability"
    )

    domain = registry.get_domain(
        domain_test
    )

    assert domain is not None, (
        "Domain lookup failed for "
        "'Economic Stability'."
    )

    print(
        "Domain lookup:         PASS"
    )

    # ----------------------------------------------------------------
    # FACTOR LOOKUP
    # ----------------------------------------------------------------

    assert registry.factors, (
        "No SDOH factors loaded."
    )

    factor_id = next(
        iter(registry.factors)
    )

    factor = registry.get_factor(
        factor_id
    )

    assert factor is not None

    print(
        "Factor lookup:         PASS"
    )

    # ----------------------------------------------------------------
    # INTERVENTION LOOKUP
    # ----------------------------------------------------------------

    assert registry.interventions, (
        "No interventions loaded."
    )

    intervention_id = next(
        iter(
            registry.interventions
        )
    )

    intervention = (
        registry.get_intervention(
            intervention_id
        )
    )

    assert intervention is not None

    print(
        "Intervention lookup:   PASS"
    )

    # ----------------------------------------------------------------
    # FACTOR CONTEXT
    # ----------------------------------------------------------------

    context = (
        registry.get_factor_context(
            factor_id
        )
    )

    assert "factor" in context
    assert "domains" in context
    assert "interventions" in context
    assert "evidence" in context

    print(
        "Factor context:        PASS"
    )

    # ----------------------------------------------------------------
    # FEATURE → FACTOR
    # ----------------------------------------------------------------

    if registry.factor_by_feature:

        feature_name = next(
            iter(
                registry.factor_by_feature
            )
        )

        resolved = (
            registry.get_factor_by_feature(
                feature_name
            )
        )

        assert resolved is not None

        print(
            "Feature → factor:      PASS"
        )

    else:

        print(
            "Feature → factor:      "
            "SKIPPED"
        )

    # ----------------------------------------------------------------
    # EVIDENCE
    # ----------------------------------------------------------------

    assert registry.evidence, (
        "Evidence records were not loaded."
    )

    print(
        "Evidence records:      PASS"
    )

    evidence_id = next(
        iter(registry.evidence)
    )

    evidence = (
        registry.get_evidence(
            evidence_id
        )
    )

    assert evidence is not None

    print(
        "Evidence lookup:       PASS"
    )

    # ----------------------------------------------------------------
    # EVIDENCE → FACTOR
    # ----------------------------------------------------------------

    evidence_factor_ids = [
        factor_id
        for factor_id, evidence_ids
        in registry.evidence_by_factor.items()
        if evidence_ids
    ]

    assert evidence_factor_ids, (
        "No evidence → factor "
        "relationships were built."
    )

    print(
        "Evidence → factor:     PASS"
    )

    # ----------------------------------------------------------------
    # EVIDENCE → INTERVENTION
    # ----------------------------------------------------------------

    if registry.evidence_by_intervention:

        print(
            "Evidence → intervention:"
            " PASS"
        )

    else:

        print(
            "Evidence → intervention:"
            " SKIPPED"
        )

    print()
    print(
        "REGISTRY SELF-TEST: PASSED"
    )
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    _run_registry_self_test()