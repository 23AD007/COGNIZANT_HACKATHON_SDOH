"""
HealthLens Knowledge Base
SDOH Intervention Knowledge

This module defines the intervention knowledge base used by the
HealthLens intervention-prioritization and knowledge-graph layers.

Architecture:

    SDOH Domain
        ↓
    SDOH Factor
        ↓
    Intervention
        ↓
    Recommended Action

Important:
- This module does NOT calculate member risk.
- This module does NOT modify the ML pipeline.
- This module does NOT assign interventions directly to members.
- It provides structured intervention knowledge that downstream
  components can use when interpreting member-specific SDOH needs.

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


# ============================================================================
# VERSION
# ============================================================================

KNOWLEDGE_BASE_VERSION = "1.0.0"


# ============================================================================
# DATA MODEL
# ============================================================================

@dataclass(frozen=True)
class Intervention:
    """
    Structured representation of an SDOH intervention.

    intervention_id:
        Stable machine-readable identifier.

    name:
        Human-readable intervention name.

    domain:
        Primary SDOH domain.

    description:
        Description of the intervention.

    actions:
        Concrete actions that may be considered.

    target_factors:
        SDOH factor identifiers that can support this intervention.

    eligibility_signals:
        General contextual signals suggesting relevance.

    expected_outcomes:
        Expected intervention outcomes.

    evidence_level:
        Knowledge-base evidence classification.

    priority:
        Default intervention priority when the intervention is relevant.
    """

    intervention_id: str
    name: str
    domain: str
    description: str
    actions: List[str]
    target_factors: List[str]
    eligibility_signals: List[str]
    expected_outcomes: List[str]
    evidence_level: str = "knowledge_base"
    priority: str = "medium"

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# INTERVENTION DEFINITIONS
# ============================================================================

INTERVENTIONS: List[Intervention] = [

    # ------------------------------------------------------------------------
    # ECONOMIC STABILITY
    # ------------------------------------------------------------------------

    Intervention(
        intervention_id="INT_ECONOMIC_BENEFITS",
        name="Financial assistance and benefits navigation",
        domain="Economic Stability",
        description=(
            "Connect members with available financial assistance, public "
            "benefits, and benefits-navigation resources."
        ),
        actions=[
            "Benefits eligibility screening",
            "SNAP application assistance",
            "Public assistance navigation",
            "Financial resource referral",
            "Benefits enrollment support",
        ],
        target_factors=[
            "snap_households_count_sum",
            "public_assistance_pct",
            "poverty_pct",
        ],
        eligibility_signals=[
            "High economic instability",
            "High SNAP concentration",
            "High poverty",
            "Public assistance need",
        ],
        expected_outcomes=[
            "Improved access to financial resources",
            "Improved benefits enrollment",
            "Reduced economic barriers to care",
        ],
        priority="high",
    ),

    # ------------------------------------------------------------------------
    # TRANSPORTATION
    # ------------------------------------------------------------------------

    Intervention(
        intervention_id="INT_TRANSPORTATION",
        name="Transportation assistance for healthcare access",
        domain="Transportation",
        description=(
            "Reduce transportation barriers that may prevent members from "
            "accessing healthcare and community resources."
        ),
        actions=[
            "Transportation resource referral",
            "Non-emergency medical transportation",
            "Community transportation referral",
            "Ride assistance",
            "Transportation scheduling support",
        ],
        target_factors=[
            "driving_low_access_population_beyond_1mi_10mi_count_sum",
            "driving_no_vehicle_households_beyond_1mi_count_sum",
            "driving_snap_households_beyond_1mi_count_sum",
            "straight_low_access_population_beyond_1mi_10mi_count_sum",
            "straight_no_vehicle_households_beyond_1mi_count_sum",
            "straight_snap_households_beyond_1mi_count_sum",
            "driving_low_income_low_access_tract_count",
            "driving_low_vehicle_access_tract_count",
            "straight_low_income_low_access_tract_count",
            "straight_low_vehicle_access_tract_count",
        ],
        eligibility_signals=[
            "Low vehicle access",
            "High transportation burden",
            "Low-access population concentration",
            "Transportation-related healthcare access barrier",
        ],
        expected_outcomes=[
            "Improved healthcare access",
            "Reduced missed appointments",
            "Improved continuity of care",
        ],
        priority="high",
    ),

    # ------------------------------------------------------------------------
    # HOUSING
    # ------------------------------------------------------------------------

    Intervention(
        intervention_id="INT_HOUSING_STABILITY",
        name="Housing stability and housing-support referral",
        domain="Housing",
        description=(
            "Connect members with housing-support resources when housing "
            "cost, crowding, or instability may create barriers to health."
        ),
        actions=[
            "Housing resource referral",
            "Rental assistance navigation",
            "Housing counseling",
            "Housing stability assessment",
            "Community housing referral",
        ],
        target_factors=[
            "housing_rent_30_to_34_9_pct",
            "housing_rent_35_plus_pct",
            "housing_crowded_1_01_to_1_50_pct",
            "housing_crowded_1_51_plus_pct",
            "housing_vacancy_pct",
            "housing_renter_pct",
            "housing_cost_burden_30pct_or_more",
        ],
        eligibility_signals=[
            "High housing cost burden",
            "High rental burden",
            "Housing overcrowding",
            "Housing instability",
        ],
        expected_outcomes=[
            "Improved housing stability",
            "Reduced housing-related stress",
            "Improved ability to maintain healthcare access",
        ],
        priority="high",
    ),

    # ------------------------------------------------------------------------
    # EDUCATION ACCESS
    # ------------------------------------------------------------------------

    Intervention(
        intervention_id="INT_EDUCATION_SUPPORT",
        name="Health education and literacy support",
        domain="Education Access",
        description=(
            "Provide accessible health education and navigation support "
            "when educational barriers may affect healthcare engagement."
        ),
        actions=[
            "Health literacy support",
            "Patient education",
            "Healthcare navigation",
            "Plain-language educational resources",
            "Community education referral",
        ],
        target_factors=[
            "education_less_than_9th_pct",
            "education_9th_to_12th_no_diploma_pct",
            "education_less_than_high_school_pct",
            "education_high_school_pct",
            "education_college_pct",
            "education_bachelors_or_higher_pct",
        ],
        eligibility_signals=[
            "Lower educational attainment",
            "Health literacy barrier",
            "Healthcare navigation difficulty",
        ],
        expected_outcomes=[
            "Improved health literacy",
            "Improved healthcare navigation",
            "Improved engagement with preventive care",
        ],
        priority="medium",
    ),

    # ------------------------------------------------------------------------
    # DIGITAL ACCESS
    # ------------------------------------------------------------------------

    Intervention(
        intervention_id="INT_DIGITAL_ACCESS",
        name="Digital-access and telehealth support",
        domain="Digital Access",
        description=(
            "Reduce digital barriers that may limit access to telehealth, "
            "health information, and digital healthcare services."
        ),
        actions=[
            "Digital access assessment",
            "Device-access referral",
            "Broadband resource referral",
            "Digital literacy support",
            "Telehealth setup assistance",
        ],
        target_factors=[
            "digital_with_computer_pct",
            "digital_with_broadband_pct",
            "digital_no_computer_pct",
            "digital_no_broadband_pct",
        ],
        eligibility_signals=[
            "Low broadband access",
            "Low computer access",
            "Digital literacy barrier",
            "Telehealth access barrier",
        ],
        expected_outcomes=[
            "Improved digital healthcare access",
            "Improved telehealth utilization",
            "Improved access to health information",
        ],
        priority="medium",
    ),

    # ------------------------------------------------------------------------
    # HEALTHCARE ACCESS
    # ------------------------------------------------------------------------

    Intervention(
        intervention_id="INT_HEALTHCARE_ACCESS",
        name="Healthcare navigation and preventive-care support",
        domain="Healthcare Access",
        description=(
            "Connect members with healthcare navigation and preventive-care "
            "resources when access or preventive-care barriers are present."
        ),
        actions=[
            "Primary-care navigation",
            "Preventive-care scheduling",
            "Care coordination",
            "Community healthcare referral",
            "Appointment navigation",
        ],
        target_factors=[
            "places_routine_checkup_pct",
            "places_cholesterol_screening_pct",
            "places_uninsured_pct",
        ],
        eligibility_signals=[
            "Low preventive-care utilization",
            "Healthcare access barrier",
            "Uninsured population concentration",
        ],
        expected_outcomes=[
            "Improved preventive-care utilization",
            "Improved healthcare access",
            "Improved continuity of care",
        ],
        priority="high",
    ),

    # ------------------------------------------------------------------------
    # NEIGHBORHOOD / BUILT ENVIRONMENT
    # ------------------------------------------------------------------------

    Intervention(
        intervention_id="INT_HEALTH_ENVIRONMENT",
        name="Community health and healthy-environment support",
        domain="Neighborhood and Built Environment",
        description=(
            "Connect members with community resources addressing "
            "environmental and neighborhood-level health barriers."
        ),
        actions=[
            "Community health resource referral",
            "Healthy-lifestyle resource referral",
            "Community wellness programs",
            "Local health-resource navigation",
            "Environmental health resource referral",
        ],
        target_factors=[
            "places_asthma_pct",
            "places_copd_pct",
            "places_diabetes_pct",
            "places_heart_disease_pct",
            "places_obesity_pct",
            "places_physical_inactivity_pct",
            "places_poor_mental_health_pct",
            "places_poor_physical_health_pct",
            "places_smoking_pct",
            "places_stroke_pct",
        ],
        eligibility_signals=[
            "Elevated neighborhood health burden",
            "High chronic-disease burden",
            "High physical inactivity",
            "Poor physical health concentration",
            "Poor mental health concentration",
        ],
        expected_outcomes=[
            "Improved connection to community resources",
            "Improved preventive health engagement",
            "Improved chronic-condition support",
        ],
        priority="medium",
    ),

    # ------------------------------------------------------------------------
    # SOCIAL / COMMUNITY CONTEXT
    # ------------------------------------------------------------------------

    Intervention(
        intervention_id="INT_SOCIAL_SUPPORT",
        name="Community and social-support referral",
        domain="Social and Community Context",
        description=(
            "Connect members with community and social-support resources "
            "when social or community barriers may affect health."
        ),
        actions=[
            "Community resource referral",
            "Social-support screening",
            "Community organization referral",
            "Peer-support referral",
            "Social-service navigation",
        ],
        target_factors=[],
        eligibility_signals=[
            "Social-support need",
            "Community resource need",
            "Social isolation indicator",
        ],
        expected_outcomes=[
            "Improved community connection",
            "Improved social support",
            "Improved access to community resources",
        ],
        priority="medium",
    ),
]


# ============================================================================
# INDEXES
# ============================================================================

INTERVENTION_BY_ID: Dict[str, Intervention] = {
    intervention.intervention_id: intervention
    for intervention in INTERVENTIONS
}


INTERVENTION_BY_DOMAIN: Dict[str, List[Intervention]] = {}

for intervention in INTERVENTIONS:
    INTERVENTION_BY_DOMAIN.setdefault(
        intervention.domain,
        [],
    ).append(intervention)


INTERVENTION_BY_FACTOR: Dict[str, List[Intervention]] = {}

for intervention in INTERVENTIONS:
    for factor in intervention.target_factors:
        INTERVENTION_BY_FACTOR.setdefault(
            factor,
            [],
        ).append(intervention)


# ============================================================================
# PUBLIC API
# ============================================================================

def get_intervention(intervention_id: str) -> Optional[Intervention]:
    """Return an intervention by stable identifier."""
    return INTERVENTION_BY_ID.get(intervention_id)


def get_interventions() -> List[Intervention]:
    """Return all interventions."""
    return list(INTERVENTIONS)


def get_interventions_for_domain(
    domain: str,
) -> List[Intervention]:
    """Return interventions associated with an SDOH domain."""
    return list(
        INTERVENTION_BY_DOMAIN.get(domain, [])
    )


def get_interventions_for_factor(
    factor: str,
) -> List[Intervention]:
    """Return interventions associated with an SDOH factor."""
    return list(
        INTERVENTION_BY_FACTOR.get(factor, [])
    )


def get_interventions_for_factors(
    factors: List[str],
) -> List[Intervention]:
    """
    Return unique interventions associated with any supplied factors.

    Ordering follows the order in which interventions are defined in
    the knowledge base.
    """

    factor_set = set(factors)

    results = []

    for intervention in INTERVENTIONS:
        if factor_set.intersection(intervention.target_factors):
            results.append(intervention)

    return results


def serialize_interventions() -> List[Dict]:
    """Serialize the intervention knowledge base."""
    return [
        intervention.to_dict()
        for intervention in INTERVENTIONS
    ]


# ============================================================================
# VALIDATION
# ============================================================================

def validate_intervention_knowledge_base() -> None:
    """Validate structural integrity of the intervention KB."""

    if not INTERVENTIONS:
        raise AssertionError(
            "Intervention knowledge base is empty."
        )

    intervention_ids = [
        intervention.intervention_id
        for intervention in INTERVENTIONS
    ]

    if len(intervention_ids) != len(set(intervention_ids)):
        raise AssertionError(
            "Duplicate intervention_id detected."
        )

    for intervention in INTERVENTIONS:

        if not intervention.intervention_id:
            raise AssertionError(
                "Intervention ID cannot be empty."
            )

        if not intervention.name:
            raise AssertionError(
                f"Intervention name missing: "
                f"{intervention.intervention_id}"
            )

        if not intervention.domain:
            raise AssertionError(
                f"Intervention domain missing: "
                f"{intervention.intervention_id}"
            )

        if not intervention.description:
            raise AssertionError(
                f"Intervention description missing: "
                f"{intervention.intervention_id}"
            )

        if not intervention.actions:
            raise AssertionError(
                f"No actions defined for: "
                f"{intervention.intervention_id}"
            )

        if not intervention.expected_outcomes:
            raise AssertionError(
                f"No expected outcomes defined for: "
                f"{intervention.intervention_id}"
            )

    # Verify indexes.
    if len(INTERVENTION_BY_ID) != len(INTERVENTIONS):
        raise AssertionError(
            "INTERVENTION_BY_ID index is inconsistent."
        )

    # Verify factor index.
    for factor, interventions in INTERVENTION_BY_FACTOR.items():
        for intervention in interventions:
            if factor not in intervention.target_factors:
                raise AssertionError(
                    "Factor index inconsistency."
                )

    # Verify domain index.
    for domain, interventions in INTERVENTION_BY_DOMAIN.items():
        for intervention in interventions:
            if intervention.domain != domain:
                raise AssertionError(
                    "Domain index inconsistency."
                )


# ============================================================================
# SELF TEST
# ============================================================================

def _run_self_test() -> None:
    print("=" * 70)
    print("HEALTHLENS KNOWLEDGE BASE — INTERVENTIONS")
    print("=" * 70)

    print(
        f"Knowledge base version: {KNOWLEDGE_BASE_VERSION}"
    )

    print(
        f"Interventions:           {len(INTERVENTIONS)}"
    )

    validate_intervention_knowledge_base()

    print()
    print("INTERVENTIONS")
    print("-" * 70)

    for intervention in INTERVENTIONS:
        print(
            f"{intervention.intervention_id:<32}"
            f"{intervention.name}"
        )

    print()
    print("DOMAIN COVERAGE")
    print("-" * 70)

    for domain, interventions in sorted(
        INTERVENTION_BY_DOMAIN.items()
    ):
        print(
            f"{domain:<40}"
            f"{len(interventions)} intervention(s)"
        )

    print()
    print("FACTOR INDEX")
    print("-" * 70)

    print(
        f"Indexed SDOH factors: "
        f"{len(INTERVENTION_BY_FACTOR)}"
    )

    print()
    print("INTERVENTION KNOWLEDGE BASE SELF-TEST: PASSED")
    print("=" * 70)


# ============================================================================
# MODULE ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    _run_self_test()