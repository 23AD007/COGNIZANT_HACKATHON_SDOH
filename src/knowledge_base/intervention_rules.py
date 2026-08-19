"""
HealthLens Knowledge Base
SDOH intervention rules.

Purpose
-------
Defines deterministic, explainable relationships between SDOH factors,
SDOH domains, and intervention recommendations.

Architecture
------------
ML model:
    Predicts member-level clinical risk.

Knowledge base:
    Encodes domain/factor/intervention knowledge.

Knowledge graph:
    Connects member-specific observations to the knowledge base.

This module does NOT:
    - train models
    - calculate member risk
    - overwrite ML predictions
    - assign counties
    - modify existing processed datasets
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


# ---------------------------------------------------------------------
# VERSION
# ---------------------------------------------------------------------

KNOWLEDGE_BASE_VERSION = "1.0.0"


# ---------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class InterventionRule:
    """
    Deterministic intervention rule.

    factor
        Model/data feature representing an SDOH signal.

    domain
        SDOH domain containing the factor.

    intervention
        Human-readable intervention recommendation.

    rationale
        Explainable reason for the recommendation.

    priority
        Relative intervention priority for this factor.

    direction
        Expected direction of concern.

    evidence_type
        Type of knowledge supporting the rule.
    """

    factor: str
    domain: str
    intervention: str
    rationale: str
    priority: str = "Medium"
    direction: str = "high"
    evidence_type: str = "SDOH association"


# ---------------------------------------------------------------------
# INTERVENTION RULES
# ---------------------------------------------------------------------

INTERVENTION_RULES: List[InterventionRule] = [

    # ================================================================
    # ECONOMIC STABILITY
    # ================================================================

    InterventionRule(
        factor="snap_households_count_sum",
        domain="Economic Stability",
        intervention="Financial assistance and benefits navigation",
        rationale=(
            "High SNAP household concentration may indicate elevated "
            "economic hardship and potential need for benefits navigation."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="poverty_pct",
        domain="Economic Stability",
        intervention="Financial assistance and benefits navigation",
        rationale=(
            "Higher poverty prevalence may indicate financial barriers "
            "that can interfere with healthcare access and stability."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="public_assistance_pct",
        domain="Economic Stability",
        intervention="Benefits eligibility and social-service navigation",
        rationale=(
            "Higher public-assistance prevalence may indicate greater "
            "economic vulnerability and need for coordinated support."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="unemployment_pct",
        domain="Economic Stability",
        intervention="Employment and financial-resource navigation",
        rationale=(
            "Higher unemployment may indicate economic instability "
            "that can contribute to unmet social needs."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="median_household_income",
        domain="Economic Stability",
        intervention="Financial resource navigation",
        rationale=(
            "Lower household income may indicate reduced financial "
            "capacity to manage healthcare and living expenses."
        ),
        priority="Medium",
        direction="low",
    ),

    # ================================================================
    # EDUCATION ACCESS
    # ================================================================

    InterventionRule(
        factor="education_less_than_9th_pct",
        domain="Education Access",
        intervention="Health-literacy and care-navigation support",
        rationale=(
            "Higher prevalence of very low educational attainment may "
            "indicate a need for simplified health information and "
            "navigation assistance."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="education_9th_to_12th_no_diploma_pct",
        domain="Education Access",
        intervention="Health-literacy and care-navigation support",
        rationale=(
            "Lower educational attainment can create barriers to "
            "understanding and navigating healthcare services."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="education_high_school_pct",
        domain="Education Access",
        intervention="Health-literacy and care-navigation support",
        rationale=(
            "Educational attainment can influence healthcare navigation "
            "and access to understandable health information."
        ),
        priority="Medium",
    ),

    # ================================================================
    # HEALTHCARE ACCESS
    # ================================================================

    InterventionRule(
        factor="places_uninsured_pct",
        domain="Healthcare Access",
        intervention="Insurance enrollment and healthcare navigation",
        rationale=(
            "Higher uninsured prevalence may indicate barriers to "
            "obtaining affordable healthcare services."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="places_routine_checkup_pct",
        domain="Healthcare Access",
        intervention="Primary-care connection and preventive-care outreach",
        rationale=(
            "Lower routine checkup prevalence may indicate reduced "
            "engagement with preventive healthcare."
        ),
        priority="High",
        direction="low",
    ),

    InterventionRule(
        factor="places_cholesterol_screening_pct",
        domain="Healthcare Access",
        intervention="Preventive-care screening navigation",
        rationale=(
            "Lower screening prevalence may indicate gaps in preventive "
            "care engagement."
        ),
        priority="Medium",
        direction="low",
    ),

    InterventionRule(
        factor="uninsured_pct",
        domain="Healthcare Access",
        intervention="Insurance enrollment and healthcare navigation",
        rationale=(
            "Uninsured prevalence may indicate financial and access "
            "barriers to healthcare."
        ),
        priority="High",
    ),

    # ================================================================
    # HOUSING
    # ================================================================

    InterventionRule(
        factor="housing_vacancy_pct",
        domain="Housing",
        intervention="Housing stability assessment",
        rationale=(
            "Elevated housing vacancy may reflect local housing "
            "instability or neighborhood-level housing conditions."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="housing_no_vehicle_pct",
        domain="Housing",
        intervention="Housing and transportation support",
        rationale=(
            "Households without vehicles may experience combined "
            "housing and transportation barriers."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="housing_renter_pct",
        domain="Housing",
        intervention="Housing stability and tenant-resource navigation",
        rationale=(
            "Higher renter prevalence may indicate greater exposure "
            "to housing-cost and housing-stability pressures."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="housing_crowded_1_01_to_1_50_pct",
        domain="Housing",
        intervention="Housing stability assessment",
        rationale=(
            "Crowded housing conditions may indicate housing instability "
            "or inadequate living space."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="housing_crowded_1_51_plus_pct",
        domain="Housing",
        intervention="Housing stability assessment",
        rationale=(
            "Severe crowding may indicate substantial housing-related "
            "social need."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="housing_rent_30_to_34_9_pct",
        domain="Housing",
        intervention="Housing affordability assistance",
        rationale=(
            "Higher rent burden may indicate reduced financial capacity "
            "after housing expenses."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="housing_rent_35_plus_pct",
        domain="Housing",
        intervention="Housing affordability assistance",
        rationale=(
            "High rent burden may indicate substantial housing-cost "
            "pressure."
        ),
        priority="High",
    ),

    # ================================================================
    # TRANSPORTATION
    # ================================================================

    InterventionRule(
        factor="households_without_vehicle_count_sum",
        domain="Transportation",
        intervention="Transportation assistance for healthcare access",
        rationale=(
            "Households without vehicles may face difficulty reaching "
            "healthcare and community services."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="driving_low_access_population_beyond_1mi_10mi_count_sum",
        domain="Transportation",
        intervention="Transportation assistance for healthcare access",
        rationale=(
            "Low-access populations may experience geographic barriers "
            "to healthcare and essential services."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="driving_no_vehicle_households_beyond_1mi_count_sum",
        domain="Transportation",
        intervention="Transportation assistance for healthcare access",
        rationale=(
            "Households without vehicles located beyond convenient "
            "service access may face transportation barriers."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="driving_snap_households_beyond_1mi_count_sum",
        domain="Transportation",
        intervention="Transportation and benefits-access support",
        rationale=(
            "Combined economic and transportation barriers may increase "
            "difficulty reaching essential services."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="driving_low_income_low_access_tract_count",
        domain="Transportation",
        intervention="Transportation assistance for healthcare access",
        rationale=(
            "Low-income areas with limited access may have increased "
            "transportation-related barriers."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="driving_low_vehicle_access_tract_count",
        domain="Transportation",
        intervention="Transportation assistance for healthcare access",
        rationale=(
            "Low vehicle access may reduce the ability to reach "
            "healthcare and social services."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="straight_no_vehicle_households_beyond_1mi_count_sum",
        domain="Transportation",
        intervention="Transportation assistance for healthcare access",
        rationale=(
            "Lack of vehicle access combined with geographic distance "
            "may create healthcare access barriers."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="straight_snap_households_beyond_1mi_count_sum",
        domain="Transportation",
        intervention="Transportation and benefits-access support",
        rationale=(
            "Economic vulnerability combined with distance may increase "
            "difficulty accessing essential services."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="straight_low_income_low_access_tract_count",
        domain="Transportation",
        intervention="Transportation assistance for healthcare access",
        rationale=(
            "Low-income and low-access areas may experience significant "
            "transportation barriers."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="straight_low_vehicle_access_tract_count",
        domain="Transportation",
        intervention="Transportation assistance for healthcare access",
        rationale=(
            "Low vehicle access may create barriers to healthcare "
            "and essential services."
        ),
        priority="High",
    ),

    # ================================================================
    # DIGITAL ACCESS
    # ================================================================

    InterventionRule(
        factor="digital_with_computer_pct",
        domain="Digital Access",
        intervention="Digital-access and telehealth support",
        rationale=(
            "Lower computer access may limit the ability to use "
            "digital healthcare resources."
        ),
        priority="Medium",
        direction="low",
    ),

    InterventionRule(
        factor="digital_with_broadband_pct",
        domain="Digital Access",
        intervention="Digital-access and telehealth support",
        rationale=(
            "Lower broadband access may limit telehealth and "
            "digital-health engagement."
        ),
        priority="High",
        direction="low",
    ),

    # ================================================================
    # NEIGHBORHOOD AND BUILT ENVIRONMENT
    # ================================================================

    InterventionRule(
        factor="places_asthma_pct",
        domain="Neighborhood and Built Environment",
        intervention="Environmental-health and chronic-condition support",
        rationale=(
            "Higher asthma prevalence may indicate increased "
            "community-level respiratory health burden."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="places_copd_pct",
        domain="Neighborhood and Built Environment",
        intervention="Respiratory-health support and care coordination",
        rationale=(
            "Higher COPD prevalence may indicate elevated respiratory "
            "health burden requiring coordinated support."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="places_diabetes_pct",
        domain="Neighborhood and Built Environment",
        intervention="Chronic-condition prevention and management support",
        rationale=(
            "Higher diabetes prevalence may indicate elevated "
            "community chronic-disease burden."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="places_heart_disease_pct",
        domain="Neighborhood and Built Environment",
        intervention="Cardiovascular-risk prevention support",
        rationale=(
            "Higher heart-disease prevalence may indicate elevated "
            "community cardiovascular burden."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="places_obesity_pct",
        domain="Neighborhood and Built Environment",
        intervention="Healthy-lifestyle and chronic-condition support",
        rationale=(
            "Higher obesity prevalence may indicate elevated "
            "community-level metabolic health burden."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="places_physical_inactivity_pct",
        domain="Neighborhood and Built Environment",
        intervention="Physical-activity and wellness support",
        rationale=(
            "Higher physical inactivity prevalence may indicate "
            "community-level barriers to healthy activity."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="places_poor_mental_health_pct",
        domain="Neighborhood and Built Environment",
        intervention="Behavioral-health navigation",
        rationale=(
            "Higher prevalence of poor mental health may indicate "
            "greater need for behavioral-health access."
        ),
        priority="High",
    ),

    InterventionRule(
        factor="places_poor_physical_health_pct",
        domain="Neighborhood and Built Environment",
        intervention="Primary-care and chronic-condition support",
        rationale=(
            "Higher poor physical health prevalence may indicate "
            "greater community health burden."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="places_smoking_pct",
        domain="Neighborhood and Built Environment",
        intervention="Smoking-cessation support",
        rationale=(
            "Higher smoking prevalence may indicate increased need "
            "for tobacco-cessation resources."
        ),
        priority="Medium",
    ),

    InterventionRule(
        factor="places_stroke_pct",
        domain="Neighborhood and Built Environment",
        intervention="Cardiovascular-risk prevention support",
        rationale=(
            "Higher stroke prevalence may indicate elevated "
            "community cardiovascular burden."
        ),
        priority="Medium",
    ),

    # ================================================================
    # COMMUTE / ACCESS
    # ================================================================

    InterventionRule(
        factor="mean_commute_minutes",
        domain="Transportation",
        intervention="Transportation and care-access support",
        rationale=(
            "Longer commute times may indicate reduced accessibility "
            "to healthcare and community resources."
        ),
        priority="Medium",
    ),
]


# ---------------------------------------------------------------------
# INDEXES
# ---------------------------------------------------------------------

_RULES_BY_FACTOR: Dict[str, InterventionRule] = {
    rule.factor: rule
    for rule in INTERVENTION_RULES
}


_RULES_BY_DOMAIN: Dict[str, List[InterventionRule]] = {}

for rule in INTERVENTION_RULES:
    _RULES_BY_DOMAIN.setdefault(rule.domain, []).append(rule)


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

def get_intervention_rule(
    factor: str,
) -> Optional[InterventionRule]:
    """
    Return the intervention rule associated with a factor.
    """

    if factor is None:
        return None

    return _RULES_BY_FACTOR.get(str(factor).strip())


def get_rules_for_domain(
    domain: str,
) -> List[InterventionRule]:
    """
    Return all intervention rules for an SDOH domain.
    """

    if domain is None:
        return []

    return list(
        _RULES_BY_DOMAIN.get(
            str(domain).strip(),
            [],
        )
    )


def get_all_rules() -> List[InterventionRule]:
    """
    Return all intervention rules.
    """

    return list(INTERVENTION_RULES)


def get_all_factors() -> List[str]:
    """
    Return factors represented in the knowledge base.
    """

    return list(_RULES_BY_FACTOR.keys())


def get_all_domains() -> List[str]:
    """
    Return SDOH domains represented by intervention rules.
    """

    return list(_RULES_BY_DOMAIN.keys())


def rule_to_dict(
    rule: InterventionRule,
) -> Dict[str, str]:
    """
    Convert a rule to a serializable dictionary.
    """

    return asdict(rule)


def get_intervention_for_factor(
    factor: str,
) -> Optional[str]:
    """
    Convenience function returning only the intervention.
    """

    rule = get_intervention_rule(factor)

    if rule is None:
        return None

    return rule.intervention


def get_rationale_for_factor(
    factor: str,
) -> Optional[str]:
    """
    Convenience function returning only the rationale.
    """

    rule = get_intervention_rule(factor)

    if rule is None:
        return None

    return rule.rationale


# ---------------------------------------------------------------------
# SELF TEST
# ---------------------------------------------------------------------

def _run_self_test() -> None:

    print("=" * 70)
    print("HEALTHLENS KNOWLEDGE BASE — INTERVENTION RULES")
    print("=" * 70)

    print(
        f"Knowledge base version: {KNOWLEDGE_BASE_VERSION}"
    )

    print(
        f"Intervention rules:     {len(INTERVENTION_RULES)}"
    )

    print(
        f"Factors covered:        {len(get_all_factors())}"
    )

    print(
        f"Domains covered:        {len(get_all_domains())}"
    )

    assert len(INTERVENTION_RULES) > 0

    # Factor lookup
    rule = get_intervention_rule(
        "snap_households_count_sum"
    )

    assert rule is not None
    assert rule.domain == "Economic Stability"
    assert (
        rule.intervention
        == "Financial assistance and benefits navigation"
    )

    print("Factor lookup:           PASS")

    # Domain lookup
    transportation_rules = get_rules_for_domain(
        "Transportation"
    )

    assert len(transportation_rules) > 0

    print("Domain lookup:           PASS")

    # Intervention lookup
    intervention = get_intervention_for_factor(
        "digital_with_broadband_pct"
    )

    assert (
        intervention
        == "Digital-access and telehealth support"
    )

    print("Intervention lookup:     PASS")

    # Rationale lookup
    rationale = get_rationale_for_factor(
        "places_poor_mental_health_pct"
    )

    assert rationale is not None
    assert len(rationale) > 0

    print("Rationale lookup:        PASS")

    # Serialization
    serialized = rule_to_dict(rule)

    assert isinstance(serialized, dict)
    assert serialized["factor"] == (
        "snap_households_count_sum"
    )

    print("Serialization:           PASS")

    print()
    print("INTERVENTION RULE SELF-TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    _run_self_test()