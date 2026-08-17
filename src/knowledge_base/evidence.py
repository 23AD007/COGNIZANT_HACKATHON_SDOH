"""
HealthLens Knowledge Base
=========================

Evidence knowledge base for SDOH factors.

This module provides structured evidence associated with canonical
SDOH factors and domains.

Architecture:

    SDOH Factor
         |
         v
    Evidence Record
         |
         +----> Intervention linkage
         |
         +----> Knowledge Base Registry
         |
         +----> Knowledge Graph

Important:
    This module does not calculate member risk.
    This module does not modify the ML model.
    This module does not assign interventions to members.

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


# ============================================================================
# VERSION
# ============================================================================

KNOWLEDGE_BASE_VERSION = "1.0.0"


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass(frozen=True)
class EvidenceRecord:
    """
    Evidence associated with a canonical SDOH factor.
    """

    factor: str
    domain: str
    evidence_type: str
    statement: str
    relevance: str
    intervention_link: bool
    source: str
    version: str


# ============================================================================
# EVIDENCE RECORDS
# ============================================================================
#
# IMPORTANT:
#
# EVIDENCE_RECORDS is intentionally a LIST.
#
# Do not change this to a dictionary unless the complete architecture is
# changed accordingly.
#
# The registry supports list/tuple/set/mapping extraction.
#
# Keep the existing evidence records in this list.
#
# The record below is the canonical structure used by the existing
# HealthLens knowledge base.
# ============================================================================


EVIDENCE_RECORDS: list[EvidenceRecord] = [

    # ========================================================================
    # ECONOMIC STABILITY
    # ========================================================================

    EvidenceRecord(
        factor="snap_households_count_sum",
        domain="EconomicStability",
        evidence_type="Social vulnerability indicator",
        statement=(
            "Higher SNAP household concentration indicates greater "
            "community-level economic vulnerability."
        ),
        relevance=(
            "Economic instability can create barriers to maintaining "
            "healthcare access and meeting basic needs."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="public_assistance_pct",
        domain="EconomicStability",
        evidence_type="Social vulnerability indicator",
        statement=(
            "A higher proportion of households receiving public assistance "
            "indicates greater economic vulnerability."
        ),
        relevance=(
            "Economic vulnerability may affect the ability to meet basic "
            "needs and maintain consistent healthcare access."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="poverty_pct",
        domain="EconomicStability",
        evidence_type="Social vulnerability indicator",
        statement=(
            "Higher poverty prevalence indicates greater community-level "
            "economic vulnerability."
        ),
        relevance=(
            "Poverty can create financial barriers to healthcare, "
            "transportation, housing, nutrition, and other basic needs."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    # ========================================================================
    # EDUCATION ACCESS
    # ========================================================================

    EvidenceRecord(
        factor="education_less_than_9th_pct",
        domain="EducationAccess",
        evidence_type="Education access indicator",
        statement=(
            "A higher proportion of adults with less than a ninth-grade "
            "education indicates lower educational attainment."
        ),
        relevance=(
            "Lower educational attainment may create barriers to health "
            "literacy, healthcare navigation, and understanding health "
            "information."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="education_9th_to_12th_no_diploma_pct",
        domain="EducationAccess",
        evidence_type="Education access indicator",
        statement=(
            "A higher proportion of adults completing ninth through "
            "twelfth grade without a diploma indicates lower educational "
            "attainment."
        ),
        relevance=(
            "Lower educational attainment can affect health literacy and "
            "ability to navigate healthcare resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="education_high_school_pct",
        domain="EducationAccess",
        evidence_type="Education access indicator",
        statement=(
            "High-school educational attainment provides an indicator of "
            "community educational resources and attainment."
        ),
        relevance=(
            "Educational attainment is associated with the ability to "
            "understand and navigate health information."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="education_college_pct",
        domain="EducationAccess",
        evidence_type="Education access indicator",
        statement=(
            "College educational attainment provides a community-level "
            "indicator of educational opportunity."
        ),
        relevance=(
            "Educational opportunity can influence health literacy, "
            "employment opportunities, and healthcare navigation."
        ),
        intervention_link=False,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="education_bachelors_or_higher_pct",
        domain="EducationAccess",
        evidence_type="Education access indicator",
        statement=(
            "A higher proportion of adults with a bachelor's degree or "
            "higher indicates greater educational attainment."
        ),
        relevance=(
            "Educational attainment provides contextual information about "
            "community-level health literacy and socioeconomic opportunity."
        ),
        intervention_link=False,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    # ========================================================================
    # HEALTHCARE ACCESS
    # ========================================================================

    EvidenceRecord(
        factor="uninsured_pct",
        domain="HealthcareAccess",
        evidence_type="Healthcare access indicator",
        statement=(
            "A higher uninsured population indicates reduced access to "
            "healthcare coverage."
        ),
        relevance=(
            "Lack of insurance can create financial barriers to preventive "
            "care, treatment, and continuity of care."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_uninsured_pct",
        domain="HealthcareAccess",
        evidence_type="Healthcare access indicator",
        statement=(
            "Higher uninsured prevalence within the community indicates "
            "greater potential healthcare access vulnerability."
        ),
        relevance=(
            "Uninsured populations may experience greater financial "
            "barriers to healthcare services."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_routine_checkup_pct",
        domain="HealthcareAccess",
        evidence_type="Preventive-care indicator",
        statement=(
            "Routine checkup prevalence provides an indicator of "
            "preventive healthcare utilization."
        ),
        relevance=(
            "Lower preventive-care utilization may indicate barriers to "
            "routine healthcare engagement."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_cholesterol_screening_pct",
        domain="HealthcareAccess",
        evidence_type="Preventive-care indicator",
        statement=(
            "Cholesterol screening prevalence provides an indicator of "
            "preventive healthcare engagement."
        ),
        relevance=(
            "Lower screening utilization may indicate barriers to "
            "preventive healthcare access."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    # ========================================================================
    # HOUSING
    # ========================================================================

    EvidenceRecord(
        factor="housing_rent_30_to_34_9_pct",
        domain="Housing",
        evidence_type="Housing burden indicator",
        statement=(
            "A higher proportion of renter households spending "
            "30 to 34.9 percent of income on rent indicates increased "
            "housing cost burden."
        ),
        relevance=(
            "Housing cost burden can reduce resources available for "
            "healthcare, transportation, food, and other basic needs."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="housing_rent_35_plus_pct",
        domain="Housing",
        evidence_type="Housing burden indicator",
        statement=(
            "A higher proportion of renter households spending 35 percent "
            "or more of income on rent indicates substantial housing cost "
            "burden."
        ),
        relevance=(
            "High housing costs can contribute to financial instability "
            "and reduce resources available for healthcare."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="housing_crowded_1_01_to_1_50_pct",
        domain="Housing",
        evidence_type="Housing stability indicator",
        statement=(
            "A higher proportion of households with moderate overcrowding "
            "indicates increased housing pressure."
        ),
        relevance=(
            "Crowded housing conditions can affect health, privacy, "
            "stress, and exposure to illness."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="housing_crowded_1_51_plus_pct",
        domain="Housing",
        evidence_type="Housing stability indicator",
        statement=(
            "A higher proportion of severely overcrowded households "
            "indicates significant housing pressure."
        ),
        relevance=(
            "Severe overcrowding may increase health risks and indicate "
            "housing instability."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="housing_vacancy_pct",
        domain="Housing",
        evidence_type="Housing environment indicator",
        statement=(
            "Housing vacancy provides contextual information about the "
            "local housing environment."
        ),
        relevance=(
            "Housing market conditions can influence neighborhood stability "
            "and access to appropriate housing."
        ),
        intervention_link=False,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="housing_renter_pct",
        domain="Housing",
        evidence_type="Housing environment indicator",
        statement=(
            "A higher renter population provides contextual information "
            "about housing tenure within a community."
        ),
        relevance=(
            "Housing tenure can provide context for housing stability and "
            "potential exposure to rental cost pressures."
        ),
        intervention_link=False,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    # ========================================================================
    # TRANSPORTATION
    # ========================================================================

    EvidenceRecord(
        factor="households_without_vehicle_count_sum",
        domain="Transportation",
        evidence_type="Transportation access indicator",
        statement=(
            "A higher number of households without a vehicle indicates "
            "greater potential transportation vulnerability."
        ),
        relevance=(
            "Limited vehicle access may make it harder to reach healthcare "
            "services, employment, food, and community resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="driving_low_access_population_beyond_1mi_10mi_count_sum",
        domain="Transportation",
        evidence_type="Transportation access indicator",
        statement=(
            "A higher low-access population indicates greater geographic "
            "transportation barriers to essential services."
        ),
        relevance=(
            "Transportation barriers may reduce access to healthcare and "
            "other essential community resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="driving_no_vehicle_households_beyond_1mi_count_sum",
        domain="Transportation",
        evidence_type="Transportation access indicator",
        statement=(
            "Households without vehicles located beyond accessible "
            "driving distances may face increased transportation barriers."
        ),
        relevance=(
            "Limited transportation options can make healthcare access "
            "and appointment attendance more difficult."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="driving_snap_households_beyond_1mi_count_sum",
        domain="Transportation",
        evidence_type="Transportation access indicator",
        statement=(
            "SNAP households located in low-access areas may face combined "
            "economic and transportation barriers."
        ),
        relevance=(
            "Combined economic and transportation barriers can compound "
            "difficulty accessing healthcare and basic resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="straight_no_vehicle_households_beyond_1mi_count_sum",
        domain="Transportation",
        evidence_type="Transportation access indicator",
        statement=(
            "Households without vehicles located beyond accessible "
            "straight-line distances may face transportation barriers."
        ),
        relevance=(
            "Transportation limitations may interfere with healthcare "
            "access and continuity of care."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="straight_snap_households_beyond_1mi_count_sum",
        domain="Transportation",
        evidence_type="Transportation access indicator",
        statement=(
            "SNAP households experiencing geographic access limitations "
            "may face combined socioeconomic and transportation barriers."
        ),
        relevance=(
            "Multiple barriers can compound difficulty accessing healthcare "
            "and community resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="driving_low_income_low_access_tract_count",
        domain="Transportation",
        evidence_type="Transportation access indicator",
        statement=(
            "A higher number of low-income, low-access census tracts "
            "indicates increased geographic transportation vulnerability."
        ),
        relevance=(
            "Low-access areas may experience reduced ability to reach "
            "healthcare and essential services."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="driving_low_vehicle_access_tract_count",
        domain="Transportation",
        evidence_type="Transportation access indicator",
        statement=(
            "A higher number of low-vehicle-access tracts indicates "
            "greater community transportation vulnerability."
        ),
        relevance=(
            "Low vehicle access can limit healthcare and essential-service "
            "access."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="straight_low_income_low_access_tract_count",
        domain="Transportation",
        evidence_type="Transportation access indicator",
        statement=(
            "Low-income, low-access areas indicate combined socioeconomic "
            "and geographic vulnerability."
        ),
        relevance=(
            "Combined socioeconomic and transportation barriers can reduce "
            "access to healthcare."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="straight_low_vehicle_access_tract_count",
        domain="Transportation",
        evidence_type="Transportation access indicator",
        statement=(
            "Low vehicle access at the tract level indicates increased "
            "transportation vulnerability."
        ),
        relevance=(
            "Limited vehicle access may create barriers to healthcare "
            "appointments and essential services."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    # ========================================================================
    # DIGITAL ACCESS
    # ========================================================================

    EvidenceRecord(
        factor="digital_with_computer_pct",
        domain="DigitalAccess",
        evidence_type="Digital access indicator",
        statement=(
            "Computer access provides an indicator of community digital "
            "connectivity."
        ),
        relevance=(
            "Limited computer access may reduce the ability to use digital "
            "healthcare services and health information."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="digital_with_broadband_pct",
        domain="DigitalAccess",
        evidence_type="Digital access indicator",
        statement=(
            "Broadband access provides an indicator of community digital "
            "connectivity."
        ),
        relevance=(
            "Limited broadband access can restrict telehealth and digital "
            "healthcare access."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    # ========================================================================
    # NEIGHBORHOOD AND BUILT ENVIRONMENT
    # ========================================================================

    EvidenceRecord(
        factor="places_asthma_pct",
        domain="NeighborhoodBuiltEnvironment",
        evidence_type="Community health indicator",
        statement=(
            "Higher asthma prevalence indicates greater community-level "
            "respiratory health burden."
        ),
        relevance=(
            "Elevated respiratory disease burden may indicate increased "
            "need for community health resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_copd_pct",
        domain="NeighborhoodBuiltEnvironment",
        evidence_type="Community health indicator",
        statement=(
            "Higher COPD prevalence indicates greater community-level "
            "respiratory disease burden."
        ),
        relevance=(
            "Respiratory disease burden may increase the need for chronic "
            "disease and community health support."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_diabetes_pct",
        domain="NeighborhoodBuiltEnvironment",
        evidence_type="Community health indicator",
        statement=(
            "Higher diabetes prevalence indicates greater community-level "
            "chronic disease burden."
        ),
        relevance=(
            "Elevated chronic disease burden may indicate increased need "
            "for prevention and community health resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_heart_disease_pct",
        domain="NeighborhoodBuiltEnvironment",
        evidence_type="Community health indicator",
        statement=(
            "Higher heart disease prevalence indicates greater community "
            "cardiovascular disease burden."
        ),
        relevance=(
            "Higher cardiovascular disease burden may increase the need "
            "for preventive and chronic disease support."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_obesity_pct",
        domain="NeighborhoodBuiltEnvironment",
        evidence_type="Community health indicator",
        statement=(
            "Higher obesity prevalence indicates greater community-level "
            "metabolic health burden."
        ),
        relevance=(
            "Elevated obesity prevalence can indicate increased need for "
            "community wellness and prevention resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_physical_inactivity_pct",
        domain="NeighborhoodBuiltEnvironment",
        evidence_type="Community health indicator",
        statement=(
            "Higher physical inactivity prevalence indicates greater "
            "community-level inactivity burden."
        ),
        relevance=(
            "Physical inactivity may contribute to chronic disease risk "
            "and indicates potential need for wellness resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_poor_mental_health_pct",
        domain="NeighborhoodBuiltEnvironment",
        evidence_type="Community health indicator",
        statement=(
            "Higher prevalence of poor mental health indicates greater "
            "community-level mental health burden."
        ),
        relevance=(
            "Community mental health burden may indicate increased need "
            "for behavioral-health and community-support resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_poor_physical_health_pct",
        domain="NeighborhoodBuiltEnvironment",
        evidence_type="Community health indicator",
        statement=(
            "Higher prevalence of poor physical health indicates greater "
            "community-level physical health burden."
        ),
        relevance=(
            "Poor physical health burden may indicate increased need for "
            "healthcare navigation and community support."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_smoking_pct",
        domain="NeighborhoodBuiltEnvironment",
        evidence_type="Community health indicator",
        statement=(
            "Higher smoking prevalence indicates greater community-level "
            "tobacco-use burden."
        ),
        relevance=(
            "Smoking burden may indicate increased need for prevention "
            "and health-behavior support."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

    EvidenceRecord(
        factor="places_stroke_pct",
        domain="NeighborhoodBuiltEnvironment",
        evidence_type="Community health indicator",
        statement=(
            "Higher stroke prevalence indicates greater community-level "
            "cardiovascular health burden."
        ),
        relevance=(
            "Higher stroke burden may indicate increased need for "
            "preventive and chronic-care resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
        version=KNOWLEDGE_BASE_VERSION,
    ),

]


# ============================================================================
# INDEXES
# ============================================================================

EVIDENCE_BY_FACTOR: dict[str, EvidenceRecord] = {
    record.factor: record
    for record in EVIDENCE_RECORDS
}


EVIDENCE_BY_DOMAIN: dict[str, list[EvidenceRecord]] = {}


for record in EVIDENCE_RECORDS:

    EVIDENCE_BY_DOMAIN.setdefault(
        record.domain,
        [],
    ).append(record)


# ============================================================================
# NORMALIZATION
# ============================================================================

def _normalize_domain(
    domain: str,
) -> str:
    """
    Normalize domain names for lookup.
    """

    return (
        str(domain)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


# ============================================================================
# PUBLIC API
# ============================================================================

def get_evidence(
    evidence_id: str,
) -> Optional[EvidenceRecord]:
    """
    Return one evidence record.

    Current EvidenceRecord does not contain an explicit evidence_id.
    The canonical factor name is therefore used as the stable lookup key.

    Both forms are supported:

        snap_households_count_sum

        evidence_snap_households_count_sum
    """

    if not evidence_id:
        return None

    normalized = str(evidence_id).strip()

    # Direct factor lookup
    record = EVIDENCE_BY_FACTOR.get(
        normalized
    )

    if record is not None:
        return record

    # Optional evidence_<factor> form
    if normalized.startswith("evidence_"):

        factor = normalized[
            len("evidence_"):
        ]

        return EVIDENCE_BY_FACTOR.get(
            factor
        )

    return None


def get_all_evidence() -> list[EvidenceRecord]:
    """
    Return all evidence records.

    EVIDENCE_RECORDS is intentionally a list.
    """

    return list(EVIDENCE_RECORDS)


def get_evidence_for_factor(
    factor: str,
) -> list[EvidenceRecord]:
    """
    Return evidence associated with a canonical SDOH factor.
    """

    if not factor:
        return []

    record = EVIDENCE_BY_FACTOR.get(
        str(factor).strip()
    )

    if record is None:
        return []

    return [record]


def get_evidence_for_domain(
    domain: str,
) -> list[EvidenceRecord]:
    """
    Return evidence associated with an SDOH domain.

    Supports:

        EconomicStability
        Economic Stability
        economic_stability
        economic-stability
    """

    if not domain:
        return []

    normalized = _normalize_domain(
        domain
    )

    results = []

    for record in EVIDENCE_RECORDS:

        if (
            _normalize_domain(record.domain)
            == normalized
        ):
            results.append(record)

    return results


def get_intervention_linked_evidence() -> list[EvidenceRecord]:
    """
    Return all evidence records linked to interventions.
    """

    return [
        record
        for record in EVIDENCE_RECORDS
        if record.intervention_link is True
    ]


def get_evidence_for_intervention(
    intervention_id: str,
) -> list[EvidenceRecord]:
    """
    Return intervention-linked evidence.

    EvidenceRecord currently stores a boolean intervention_link rather
    than an explicit intervention_id.

    The argument is therefore accepted for API compatibility and future
    extension, while current behavior returns all evidence that is
    intervention-linked.
    """

    if not intervention_id:
        return []

    return get_intervention_linked_evidence()


def serialize_evidence() -> list[dict]:
    """
    Serialize all evidence records.
    """

    return [
        asdict(record)
        for record in EVIDENCE_RECORDS
    ]


# ============================================================================
# VALIDATION
# ============================================================================

def validate_evidence() -> None:
    """
    Validate structural integrity of the evidence knowledge base.

    Cross-module factor validation is performed by registry.py.
    """

    if not EVIDENCE_RECORDS:
        raise AssertionError(
            "Evidence knowledge base is empty."
        )

    seen_factors: set[str] = set()

    for record in EVIDENCE_RECORDS:

        # --------------------------------------------------------------
        # FACTOR
        # --------------------------------------------------------------

        if not record.factor:
            raise AssertionError(
                "Evidence record has no factor."
            )

        factor = str(
            record.factor
        ).strip()

        if factor in seen_factors:
            raise AssertionError(
                f"Duplicate evidence factor detected: "
                f"{factor}"
            )

        seen_factors.add(factor)

        # --------------------------------------------------------------
        # DOMAIN
        # --------------------------------------------------------------

        if not record.domain:
            raise AssertionError(
                f"Evidence has no domain: "
                f"{factor}"
            )

        # --------------------------------------------------------------
        # EVIDENCE TYPE
        # --------------------------------------------------------------

        if not record.evidence_type:
            raise AssertionError(
                f"Evidence has no evidence type: "
                f"{factor}"
            )

        # --------------------------------------------------------------
        # STATEMENT
        # --------------------------------------------------------------

        if not record.statement:
            raise AssertionError(
                f"Evidence has no statement: "
                f"{factor}"
            )

        # --------------------------------------------------------------
        # RELEVANCE
        # --------------------------------------------------------------

        if not record.relevance:
            raise AssertionError(
                f"Evidence has no relevance: "
                f"{factor}"
            )

        # --------------------------------------------------------------
        # SOURCE
        # --------------------------------------------------------------

        if not record.source:
            raise AssertionError(
                f"Evidence has no source: "
                f"{factor}"
            )

        # --------------------------------------------------------------
        # VERSION
        # --------------------------------------------------------------

        if not record.version:
            raise AssertionError(
                f"Evidence has no version: "
                f"{factor}"
            )

        # --------------------------------------------------------------
        # INTERVENTION LINK
        # --------------------------------------------------------------

        if not isinstance(
            record.intervention_link,
            bool,
        ):
            raise AssertionError(
                f"Invalid intervention_link for: "
                f"{factor}"
            )


# ============================================================================
# SELF TEST
# ============================================================================

def _run_evidence_self_test() -> None:

    print("=" * 70)
    print("HEALTHLENS KNOWLEDGE BASE — EVIDENCE")
    print("=" * 70)

    # ========================================================================
    # VALIDATION
    # ========================================================================

    validate_evidence()

    print(
        f"Knowledge base version: "
        f"{KNOWLEDGE_BASE_VERSION}"
    )

    print(
        f"Evidence records:       "
        f"{len(EVIDENCE_RECORDS)}"
    )

    factors = {
        record.factor
        for record in EVIDENCE_RECORDS
    }

    domains = {
        record.domain
        for record in EVIDENCE_RECORDS
    }

    print(
        f"Factors covered:        "
        f"{len(factors)}"
    )

    print(
        f"Domains covered:        "
        f"{len(domains)}"
    )

    # ========================================================================
    # FACTOR LOOKUP
    # ========================================================================

    assert factors, (
        "No evidence factors found."
    )

    first_factor = sorted(
        factors
    )[0]

    factor_results = (
        get_evidence_for_factor(
            first_factor
        )
    )

    assert factor_results, (
        f"Evidence lookup failed for factor: "
        f"{first_factor}"
    )

    assert (
        factor_results[0].factor
        == first_factor
    )

    print(
        "Factor lookup:          PASS"
    )

    # ========================================================================
    # EVIDENCE STATEMENT
    # ========================================================================

    assert all(
        isinstance(
            record.statement,
            str,
        )
        and record.statement.strip()
        for record in EVIDENCE_RECORDS
    )

    print(
        "Evidence statement:     PASS"
    )

    # ========================================================================
    # EVIDENCE RELEVANCE
    # ========================================================================

    assert all(
        isinstance(
            record.relevance,
            str,
        )
        and record.relevance.strip()
        for record in EVIDENCE_RECORDS
    )

    print(
        "Evidence relevance:     PASS"
    )

    # ========================================================================
    # DOMAIN LOOKUP
    # ========================================================================

    assert domains, (
        "No evidence domains found."
    )

    first_domain = sorted(
        domains
    )[0]

    domain_results = (
        get_evidence_for_domain(
            first_domain
        )
    )

    assert domain_results, (
        f"Domain lookup failed for: "
        f"{first_domain}"
    )

    assert all(
        _normalize_domain(record.domain)
        == _normalize_domain(first_domain)
        for record in domain_results
    )

    print(
        "Domain lookup:          PASS"
    )

    # ========================================================================
    # DIRECT EVIDENCE LOOKUP
    # ========================================================================

    direct_lookup = get_evidence(
        first_factor
    )

    assert direct_lookup is not None

    assert (
        direct_lookup.factor
        == first_factor
    )

    print(
        "Evidence lookup:        PASS"
    )

    # ========================================================================
    # PREFixed EVIDENCE LOOKUP
    # ========================================================================

    prefixed_lookup = get_evidence(
        f"evidence_{first_factor}"
    )

    assert prefixed_lookup is not None

    assert (
        prefixed_lookup.factor
        == first_factor
    )

    print(
        "Evidence ID lookup:     PASS"
    )

    # ========================================================================
    # ALL EVIDENCE
    # ========================================================================

    all_evidence = (
        get_all_evidence()
    )

    assert len(all_evidence) == len(
        EVIDENCE_RECORDS
    )

    print(
        "All evidence retrieval: PASS"
    )

    # ========================================================================
    # INTERVENTION LINKAGE
    # ========================================================================

    linked_evidence = (
        get_intervention_linked_evidence()
    )

    assert linked_evidence, (
        "No intervention-linked evidence found."
    )

    assert all(
        record.intervention_link is True
        for record in linked_evidence
    )

    print(
        "Intervention linkage:   PASS"
    )

    # ========================================================================
    # INTERVENTION API
    # ========================================================================

    intervention_evidence = (
        get_evidence_for_intervention(
            "INT_ECONOMIC_BENEFITS"
        )
    )

    assert intervention_evidence, (
        "Intervention evidence lookup failed."
    )

    print(
        "Intervention lookup:    PASS"
    )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    serialized = serialize_evidence()

    assert len(serialized) == len(
        EVIDENCE_RECORDS
    )

    assert all(
        isinstance(
            item,
            dict,
        )
        for item in serialized
    )

    print(
        "Serialization:          PASS"
    )

    # ========================================================================
    # SERIALIZATION FIELDS
    # ========================================================================

    required_fields = {
        "factor",
        "domain",
        "evidence_type",
        "statement",
        "relevance",
        "intervention_link",
        "source",
        "version",
    }

    for item in serialized:

        assert required_fields.issubset(
            item.keys()
        )

    print(
        "Serialization fields:   PASS"
    )

    # ========================================================================
    # INDEX CONSISTENCY
    # ========================================================================

    assert len(
        EVIDENCE_BY_FACTOR
    ) == len(
        EVIDENCE_RECORDS
    )

    for record in EVIDENCE_RECORDS:

        assert (
            EVIDENCE_BY_FACTOR[
                record.factor
            ]
            == record
        )

    print(
        "Index consistency:      PASS"
    )

    # ========================================================================
    # FINAL
    # ========================================================================

    print()
    print(
        "EVIDENCE SELF-TEST: PASSED"
    )
    print("=" * 70)


# ============================================================================
# BACKWARD-COMPATIBLE SELF-TEST NAME
# ============================================================================

_self_test = _run_evidence_self_test


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    _run_evidence_self_test()