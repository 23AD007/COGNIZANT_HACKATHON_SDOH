"""
HealthLens Knowledge Base
SDOH evidence definitions.

Purpose
-------
Stores structured, explainable evidence metadata associated with
SDOH factors and intervention rules.

Architecture
------------
ML model
    -> predicts member clinical risk

Knowledge Base
    -> defines SDOH domains
    -> defines SDOH factors
    -> defines intervention rules
    -> defines evidence metadata

Knowledge Graph
    -> connects member-specific observations to this knowledge

This module is intentionally deterministic.
It does not calculate member risk and does not modify model outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------
# VERSION
# ---------------------------------------------------------------------

KNOWLEDGE_BASE_VERSION = "1.0.0"


# ---------------------------------------------------------------------
# DATA MODEL
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceRecord:
    """
    Structured evidence associated with an SDOH factor.

    factor
        Exact feature name used by the HealthLens pipeline.

    domain
        SDOH domain associated with the factor.

    evidence_type
        Classification of the evidence.

    statement
        Explainable interpretation of the factor.

    relevance
        Why the factor matters to healthcare/social needs.

    intervention_link
        Whether an intervention rule exists for this factor.

    source
        Source category used by the knowledge base.

    version
        Knowledge-base version.
    """

    factor: str
    domain: str
    evidence_type: str
    statement: str
    relevance: str
    intervention_link: bool
    source: str
    version: str = KNOWLEDGE_BASE_VERSION


# ---------------------------------------------------------------------
# EVIDENCE RECORDS
# ---------------------------------------------------------------------

EVIDENCE_RECORDS: List[EvidenceRecord] = [

    # ================================================================
    # ECONOMIC STABILITY
    # ================================================================

    EvidenceRecord(
        factor="snap_households_count_sum",
        domain="Economic Stability",
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
    ),

    EvidenceRecord(
        factor="poverty_pct",
        domain="Economic Stability",
        evidence_type="Social vulnerability indicator",
        statement=(
            "Higher poverty prevalence indicates greater "
            "community-level economic hardship."
        ),
        relevance=(
            "Financial constraints may affect healthcare access, "
            "medication affordability, transportation, and housing."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="public_assistance_pct",
        domain="Economic Stability",
        evidence_type="Social vulnerability indicator",
        statement=(
            "Public-assistance prevalence provides a signal of "
            "economic vulnerability within the community."
        ),
        relevance=(
            "Members in economically vulnerable communities may "
            "benefit from coordinated benefits navigation."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="unemployment_pct",
        domain="Economic Stability",
        evidence_type="Economic indicator",
        statement=(
            "Higher unemployment may indicate reduced economic "
            "stability within the community."
        ),
        relevance=(
            "Economic instability can contribute to unmet social "
            "needs and difficulty accessing healthcare."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="median_household_income",
        domain="Economic Stability",
        evidence_type="Economic indicator",
        statement=(
            "Lower median household income indicates reduced "
            "community-level financial resources."
        ),
        relevance=(
            "Lower financial resources may increase barriers to "
            "healthcare and essential services."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    # ================================================================
    # EDUCATION ACCESS
    # ================================================================

    EvidenceRecord(
        factor="education_less_than_9th_pct",
        domain="Education Access",
        evidence_type="Education indicator",
        statement=(
            "Higher prevalence of very low educational attainment "
            "indicates an education-access vulnerability."
        ),
        relevance=(
            "Educational barriers can affect health literacy, "
            "care navigation, and understanding of health information."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="education_9th_to_12th_no_diploma_pct",
        domain="Education Access",
        evidence_type="Education indicator",
        statement=(
            "Higher prevalence of adults without a high-school "
            "diploma indicates an education-access challenge."
        ),
        relevance=(
            "Lower educational attainment may create barriers to "
            "health information and healthcare navigation."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="education_high_school_pct",
        domain="Education Access",
        evidence_type="Education indicator",
        statement=(
            "Educational attainment provides contextual information "
            "about community education access."
        ),
        relevance=(
            "Education context can inform health-literacy and "
            "care-navigation strategies."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    # ================================================================
    # HEALTHCARE ACCESS
    # ================================================================

    EvidenceRecord(
        factor="places_uninsured_pct",
        domain="Healthcare Access",
        evidence_type="Healthcare access indicator",
        statement=(
            "Higher uninsured prevalence indicates increased "
            "community-level insurance coverage barriers."
        ),
        relevance=(
            "Insurance barriers may reduce access to affordable "
            "preventive and treatment services."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="uninsured_pct",
        domain="Healthcare Access",
        evidence_type="Healthcare access indicator",
        statement=(
            "Higher uninsured prevalence indicates increased "
            "risk of healthcare access barriers."
        ),
        relevance=(
            "Insurance coverage can influence affordability and "
            "utilization of healthcare services."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_routine_checkup_pct",
        domain="Healthcare Access",
        evidence_type="Preventive-care indicator",
        statement=(
            "Lower routine-checkup prevalence may indicate reduced "
            "preventive-care engagement."
        ),
        relevance=(
            "Reduced preventive-care engagement may create opportunities "
            "for primary-care connection and outreach."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_cholesterol_screening_pct",
        domain="Healthcare Access",
        evidence_type="Preventive-care indicator",
        statement=(
            "Lower cholesterol-screening prevalence may indicate "
            "reduced preventive screening engagement."
        ),
        relevance=(
            "Preventive screening gaps may identify opportunities "
            "for healthcare navigation."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    # ================================================================
    # HOUSING
    # ================================================================

    EvidenceRecord(
        factor="housing_vacancy_pct",
        domain="Housing",
        evidence_type="Housing indicator",
        statement=(
            "Housing vacancy provides contextual information about "
            "local housing conditions."
        ),
        relevance=(
            "Housing conditions can influence stability and the "
            "social environment in which members live."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="housing_no_vehicle_pct",
        domain="Housing",
        evidence_type="Housing and transportation indicator",
        statement=(
            "Higher prevalence of households without vehicles "
            "indicates potential mobility constraints."
        ),
        relevance=(
            "Mobility constraints can affect access to healthcare, "
            "employment, food, and other essential services."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="housing_renter_pct",
        domain="Housing",
        evidence_type="Housing indicator",
        statement=(
            "Higher renter prevalence provides contextual information "
            "about housing tenure."
        ),
        relevance=(
            "Housing tenure can be associated with housing stability "
            "and affordability considerations."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="housing_crowded_1_01_to_1_50_pct",
        domain="Housing",
        evidence_type="Housing indicator",
        statement=(
            "Higher prevalence of moderately crowded housing indicates "
            "potential housing-capacity pressure."
        ),
        relevance=(
            "Crowded housing can affect living conditions and "
            "household stability."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="housing_crowded_1_51_plus_pct",
        domain="Housing",
        evidence_type="Housing indicator",
        statement=(
            "Higher prevalence of severely crowded housing indicates "
            "substantial housing-capacity pressure."
        ),
        relevance=(
            "Severe crowding may indicate significant housing-related "
            "social needs."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="housing_rent_30_to_34_9_pct",
        domain="Housing",
        evidence_type="Housing affordability indicator",
        statement=(
            "Higher moderate rent burden indicates increased "
            "housing-cost pressure."
        ),
        relevance=(
            "Housing costs can reduce resources available for "
            "healthcare and other basic needs."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="housing_rent_35_plus_pct",
        domain="Housing",
        evidence_type="Housing affordability indicator",
        statement=(
            "Higher severe rent burden indicates substantial "
            "housing-cost pressure."
        ),
        relevance=(
            "High housing costs may increase financial instability "
            "and competing social needs."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    # ================================================================
    # TRANSPORTATION
    # ================================================================

    EvidenceRecord(
        factor="households_without_vehicle_count_sum",
        domain="Transportation",
        evidence_type="Transportation indicator",
        statement=(
            "A higher number of households without vehicles indicates "
            "greater potential transportation vulnerability."
        ),
        relevance=(
            "Transportation barriers may make healthcare and social "
            "services difficult to reach."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="driving_low_access_population_beyond_1mi_10mi_count_sum",
        domain="Transportation",
        evidence_type="Geographic access indicator",
        statement=(
            "Low-access populations beyond defined geographic "
            "thresholds indicate potential transportation barriers."
        ),
        relevance=(
            "Distance and transportation constraints can reduce "
            "access to healthcare and essential services."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="driving_no_vehicle_households_beyond_1mi_count_sum",
        domain="Transportation",
        evidence_type="Transportation indicator",
        statement=(
            "Households without vehicles located beyond convenient "
            "access points indicate elevated mobility constraints."
        ),
        relevance=(
            "Combined distance and vehicle limitations may make "
            "healthcare access difficult."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="driving_snap_households_beyond_1mi_count_sum",
        domain="Transportation",
        evidence_type="Combined SDOH indicator",
        statement=(
            "The combination of SNAP households and transportation "
            "distance indicates overlapping economic and mobility needs."
        ),
        relevance=(
            "Members may face simultaneous transportation and "
            "economic barriers."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="driving_low_income_low_access_tract_count",
        domain="Transportation",
        evidence_type="Geographic vulnerability indicator",
        statement=(
            "Low-income and low-access tracts indicate overlapping "
            "economic and geographic vulnerability."
        ),
        relevance=(
            "Combined vulnerabilities can make healthcare and "
            "essential services harder to reach."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="driving_low_vehicle_access_tract_count",
        domain="Transportation",
        evidence_type="Transportation indicator",
        statement=(
            "Tracts with low vehicle access indicate potential "
            "community-level mobility barriers."
        ),
        relevance=(
            "Reduced vehicle access can constrain healthcare "
            "and social-service utilization."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="straight_no_vehicle_households_beyond_1mi_count_sum",
        domain="Transportation",
        evidence_type="Transportation indicator",
        statement=(
            "Households without vehicles and beyond defined access "
            "distances indicate potential transportation barriers."
        ),
        relevance=(
            "Distance combined with lack of vehicle access may "
            "reduce healthcare accessibility."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="straight_snap_households_beyond_1mi_count_sum",
        domain="Transportation",
        evidence_type="Combined SDOH indicator",
        statement=(
            "SNAP households beyond access distances may indicate "
            "combined economic and geographic vulnerability."
        ),
        relevance=(
            "Members may face multiple barriers to accessing "
            "healthcare and essential resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="straight_low_income_low_access_tract_count",
        domain="Transportation",
        evidence_type="Geographic vulnerability indicator",
        statement=(
            "Low-income low-access tracts indicate combined "
            "economic and geographic barriers."
        ),
        relevance=(
            "Combined barriers may reduce access to healthcare "
            "and community resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="straight_low_vehicle_access_tract_count",
        domain="Transportation",
        evidence_type="Transportation indicator",
        statement=(
            "Low vehicle access indicates potential mobility "
            "limitations at the community level."
        ),
        relevance=(
            "Mobility limitations can reduce access to healthcare "
            "and other essential services."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    # ================================================================
    # DIGITAL ACCESS
    # ================================================================

    EvidenceRecord(
        factor="digital_with_computer_pct",
        domain="Digital Access",
        evidence_type="Digital access indicator",
        statement=(
            "Lower computer access indicates a potential digital "
            "access limitation."
        ),
        relevance=(
            "Limited computer access may reduce ability to use "
            "digital healthcare resources."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="digital_with_broadband_pct",
        domain="Digital Access",
        evidence_type="Digital access indicator",
        statement=(
            "Lower broadband access indicates a potential digital "
            "connectivity limitation."
        ),
        relevance=(
            "Limited broadband access may reduce telehealth and "
            "digital-health engagement."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    # ================================================================
    # NEIGHBORHOOD AND BUILT ENVIRONMENT
    # ================================================================

    EvidenceRecord(
        factor="places_asthma_pct",
        domain="Neighborhood and Built Environment",
        evidence_type="Community health indicator",
        statement=(
            "Higher asthma prevalence indicates elevated "
            "community respiratory-health burden."
        ),
        relevance=(
            "Community respiratory burden can provide context "
            "for member-level health needs."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_copd_pct",
        domain="Neighborhood and Built Environment",
        evidence_type="Community health indicator",
        statement=(
            "Higher COPD prevalence indicates elevated "
            "community respiratory-health burden."
        ),
        relevance=(
            "Respiratory disease burden can inform care coordination "
            "and environmental-health support."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_diabetes_pct",
        domain="Neighborhood and Built Environment",
        evidence_type="Community health indicator",
        statement=(
            "Higher diabetes prevalence indicates elevated "
            "community metabolic-health burden."
        ),
        relevance=(
            "Community chronic-disease burden can provide context "
            "for prevention and disease-management needs."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_heart_disease_pct",
        domain="Neighborhood and Built Environment",
        evidence_type="Community health indicator",
        statement=(
            "Higher heart-disease prevalence indicates elevated "
            "community cardiovascular burden."
        ),
        relevance=(
            "Cardiovascular burden can inform preventive and "
            "care-coordination strategies."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_obesity_pct",
        domain="Neighborhood and Built Environment",
        evidence_type="Community health indicator",
        statement=(
            "Higher obesity prevalence indicates elevated "
            "community metabolic-health burden."
        ),
        relevance=(
            "Community metabolic-health context can support "
            "wellness and chronic-condition interventions."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_physical_inactivity_pct",
        domain="Neighborhood and Built Environment",
        evidence_type="Community health indicator",
        statement=(
            "Higher physical-inactivity prevalence indicates "
            "potential community wellness barriers."
        ),
        relevance=(
            "Physical inactivity may inform wellness and "
            "preventive-health interventions."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_poor_mental_health_pct",
        domain="Neighborhood and Built Environment",
        evidence_type="Community behavioral-health indicator",
        statement=(
            "Higher poor-mental-health prevalence indicates "
            "elevated community behavioral-health burden."
        ),
        relevance=(
            "Community behavioral-health burden can inform "
            "behavioral-health navigation."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_poor_physical_health_pct",
        domain="Neighborhood and Built Environment",
        evidence_type="Community health indicator",
        statement=(
            "Higher poor-physical-health prevalence indicates "
            "elevated community health burden."
        ),
        relevance=(
            "Community health burden can provide contextual "
            "information for care coordination."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_smoking_pct",
        domain="Neighborhood and Built Environment",
        evidence_type="Community health indicator",
        statement=(
            "Higher smoking prevalence indicates elevated "
            "community tobacco-use burden."
        ),
        relevance=(
            "Smoking burden can inform smoking-cessation "
            "and preventive-health support."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    EvidenceRecord(
        factor="places_stroke_pct",
        domain="Neighborhood and Built Environment",
        evidence_type="Community health indicator",
        statement=(
            "Higher stroke prevalence indicates elevated "
            "community cardiovascular burden."
        ),
        relevance=(
            "Cardiovascular burden can inform preventive and "
            "care-coordination strategies."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),

    # ================================================================
    # TRANSPORTATION / COMMUTE
    # ================================================================

    EvidenceRecord(
        factor="mean_commute_minutes",
        domain="Transportation",
        evidence_type="Transportation indicator",
        statement=(
            "Longer average commute times provide contextual "
            "information about transportation burden."
        ),
        relevance=(
            "Longer travel requirements may create barriers to "
            "healthcare and social-service access."
        ),
        intervention_link=True,
        source="SDOH knowledge base",
    ),
]


# ---------------------------------------------------------------------
# INDEXES
# ---------------------------------------------------------------------

_EVIDENCE_BY_FACTOR: Dict[str, EvidenceRecord] = {
    record.factor: record
    for record in EVIDENCE_RECORDS
}


_EVIDENCE_BY_DOMAIN: Dict[str, List[EvidenceRecord]] = {}

for record in EVIDENCE_RECORDS:
    _EVIDENCE_BY_DOMAIN.setdefault(
        record.domain,
        [],
    ).append(record)


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

def get_evidence(
    factor: str,
) -> Optional[EvidenceRecord]:
    """
    Return evidence associated with an SDOH factor.
    """

    if factor is None:
        return None

    return _EVIDENCE_BY_FACTOR.get(
        str(factor).strip()
    )


def get_evidence_for_domain(
    domain: str,
) -> List[EvidenceRecord]:
    """
    Return all evidence records associated with a domain.
    """

    if domain is None:
        return []

    return list(
        _EVIDENCE_BY_DOMAIN.get(
            str(domain).strip(),
            [],
        )
    )


def get_all_evidence() -> List[EvidenceRecord]:
    """
    Return all evidence records.
    """

    return list(EVIDENCE_RECORDS)


def get_evidence_statement(
    factor: str,
) -> Optional[str]:
    """
    Return only the evidence statement.
    """

    record = get_evidence(factor)

    if record is None:
        return None

    return record.statement


def get_evidence_relevance(
    factor: str,
) -> Optional[str]:
    """
    Return only the relevance explanation.
    """

    record = get_evidence(factor)

    if record is None:
        return None

    return record.relevance


def evidence_to_dict(
    record: EvidenceRecord,
) -> Dict[str, object]:
    """
    Serialize an evidence record.
    """

    return asdict(record)


def get_supported_factors() -> List[str]:
    """
    Return all factors with evidence definitions.
    """

    return list(_EVIDENCE_BY_FACTOR.keys())


def get_supported_domains() -> List[str]:
    """
    Return all domains represented in the evidence layer.
    """

    return list(_EVIDENCE_BY_DOMAIN.keys())


# ---------------------------------------------------------------------
# SELF TEST
# ---------------------------------------------------------------------

def _run_self_test() -> None:

    print("=" * 70)
    print("HEALTHLENS KNOWLEDGE BASE — EVIDENCE")
    print("=" * 70)

    print(
        f"Knowledge base version: {KNOWLEDGE_BASE_VERSION}"
    )

    print(
        f"Evidence records:       {len(EVIDENCE_RECORDS)}"
    )

    print(
        f"Factors covered:        {len(get_supported_factors())}"
    )

    print(
        f"Domains covered:        {len(get_supported_domains())}"
    )

    assert len(EVIDENCE_RECORDS) > 0

    # ---------------------------------------------------------------
    # Factor lookup
    # ---------------------------------------------------------------

    record = get_evidence(
        "snap_households_count_sum"
    )

    assert record is not None

    assert (
        record.domain
        == "Economic Stability"
    )

    print("Factor lookup:           PASS")

    # ---------------------------------------------------------------
    # Statement
    # ---------------------------------------------------------------

    statement = get_evidence_statement(
        "places_poor_mental_health_pct"
    )

    assert statement is not None
    assert len(statement) > 0

    print("Evidence statement:      PASS")

    # ---------------------------------------------------------------
    # Relevance
    # ---------------------------------------------------------------

    relevance = get_evidence_relevance(
        "digital_with_broadband_pct"
    )

    assert relevance is not None
    assert len(relevance) > 0

    print("Evidence relevance:      PASS")

    # ---------------------------------------------------------------
    # Domain lookup
    # ---------------------------------------------------------------

    transportation = get_evidence_for_domain(
        "Transportation"
    )

    assert len(transportation) > 0

    print("Domain lookup:           PASS")

    # ---------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------

    serialized = evidence_to_dict(record)

    assert isinstance(serialized, dict)

    assert (
        serialized["factor"]
        == "snap_households_count_sum"
    )

    print("Serialization:           PASS")

    # ---------------------------------------------------------------
    # Intervention linkage
    # ---------------------------------------------------------------

    assert record.intervention_link is True

    print("Intervention linkage:    PASS")

    print()
    print("EVIDENCE SELF-TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_test()