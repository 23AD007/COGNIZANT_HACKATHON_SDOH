"""
HealthLens SDOH factor knowledge base.

This module defines the canonical relationship:

    model feature -> SDOH factor -> SDOH domain

It does NOT contain member-specific values.

Member-specific values continue to come from:

    data/processed/member_model_features.csv

The Knowledge Graph can then connect an individual member's
observed factor values to these canonical definitions.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SDOHFactor:
    """
    Canonical SDOH factor definition.
    """

    key: str
    name: str
    domain_key: str
    description: str
    direction: str
    intervention: str


# ---------------------------------------------------------------------
# CANONICAL FACTORS
# ---------------------------------------------------------------------

SDOH_FACTORS: Dict[str, SDOHFactor] = {

    # ================================================================
    # ECONOMIC STABILITY
    # ================================================================

    "snap_households_count_sum": SDOHFactor(
        key="snap_households_count_sum",
        name="SNAP household concentration",
        domain_key="economic_stability",
        description=(
            "Number of households receiving SNAP assistance "
            "within the relevant geographic area."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Financial assistance and benefits navigation"
        ),
    ),

    "public_assistance_pct": SDOHFactor(
        key="public_assistance_pct",
        name="Public assistance prevalence",
        domain_key="economic_stability",
        description=(
            "Percentage of households receiving public assistance."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Financial assistance and benefits navigation"
        ),
    ),

    "poverty_pct": SDOHFactor(
        key="poverty_pct",
        name="Poverty prevalence",
        domain_key="economic_stability",
        description=(
            "Percentage of the population living below the "
            "defined poverty threshold."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Financial assistance and benefits navigation"
        ),
    ),

    "unemployment_pct": SDOHFactor(
        key="unemployment_pct",
        name="Unemployment prevalence",
        domain_key="economic_stability",
        description=(
            "Percentage of the labor force that is unemployed."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Employment and financial-support navigation"
        ),
    ),

    "median_household_income": SDOHFactor(
        key="median_household_income",
        name="Median household income",
        domain_key="economic_stability",
        description=(
            "Median household income in the member's geographic area."
        ),
        direction="lower_is_greater_need",
        intervention=(
            "Financial assistance and benefits navigation"
        ),
    ),

    # ================================================================
    # EDUCATION ACCESS
    # ================================================================

    "education_less_than_9th_pct": SDOHFactor(
        key="education_less_than_9th_pct",
        name="Very low educational attainment",
        domain_key="education_access",
        description=(
            "Percentage of adults with less than a ninth-grade "
            "education."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Health-literacy and education support"
        ),
    ),

    "education_9th_to_12th_no_diploma_pct": SDOHFactor(
        key="education_9th_to_12th_no_diploma_pct",
        name="No high-school diploma",
        domain_key="education_access",
        description=(
            "Percentage of adults completing grades 9–12 without "
            "receiving a high-school diploma."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Health-literacy and education support"
        ),
    ),

    "education_high_school_pct": SDOHFactor(
        key="education_high_school_pct",
        name="High-school educational attainment",
        domain_key="education_access",
        description=(
            "Percentage of adults whose highest educational "
            "attainment is high school."
        ),
        direction="contextual",
        intervention=(
            "Health-literacy and education support"
        ),
    ),

    "education_college_pct": SDOHFactor(
        key="education_college_pct",
        name="College educational attainment",
        domain_key="education_access",
        description=(
            "Percentage of adults with college-level educational "
            "attainment."
        ),
        direction="contextual",
        intervention=(
            "Health-literacy and education support"
        ),
    ),

    "education_bachelors_or_higher_pct": SDOHFactor(
        key="education_bachelors_or_higher_pct",
        name="Bachelor's degree or higher",
        domain_key="education_access",
        description=(
            "Percentage of adults with a bachelor's degree or higher."
        ),
        direction="contextual",
        intervention=(
            "Health-literacy and education support"
        ),
    ),

    # ================================================================
    # HEALTHCARE ACCESS
    # ================================================================

    "uninsured_pct": SDOHFactor(
        key="uninsured_pct",
        name="Uninsured population",
        domain_key="healthcare_access",
        description=(
            "Percentage of the population without health insurance."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Insurance and healthcare-access navigation"
        ),
    ),

    "places_uninsured_pct": SDOHFactor(
        key="places_uninsured_pct",
        name="PLACES uninsured prevalence",
        domain_key="healthcare_access",
        description=(
            "County-level uninsured prevalence from CDC PLACES."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Insurance and healthcare-access navigation"
        ),
    ),

    "places_routine_checkup_pct": SDOHFactor(
        key="places_routine_checkup_pct",
        name="Routine checkup prevalence",
        domain_key="healthcare_access",
        description=(
            "Population-level prevalence related to routine "
            "healthcare checkups."
        ),
        direction="lower_is_greater_need",
        intervention=(
            "Primary-care engagement and preventive-care navigation"
        ),
    ),

    "places_cholesterol_screening_pct": SDOHFactor(
        key="places_cholesterol_screening_pct",
        name="Cholesterol screening prevalence",
        domain_key="healthcare_access",
        description=(
            "Population-level prevalence related to cholesterol screening."
        ),
        direction="lower_is_greater_need",
        intervention=(
            "Preventive-care and screening navigation"
        ),
    ),

    # ================================================================
    # HOUSING
    # ================================================================

    "housing_vacancy_pct": SDOHFactor(
        key="housing_vacancy_pct",
        name="Housing vacancy",
        domain_key="housing",
        description=(
            "Percentage of housing units that are vacant."
        ),
        direction="contextual",
        intervention=(
            "Housing-stability support"
        ),
    ),

    "housing_no_vehicle_pct": SDOHFactor(
        key="housing_no_vehicle_pct",
        name="Households without vehicles",
        domain_key="housing",
        description=(
            "Percentage of households without access to a vehicle."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation assistance"
        ),
    ),

    "housing_renter_pct": SDOHFactor(
        key="housing_renter_pct",
        name="Renter prevalence",
        domain_key="housing",
        description=(
            "Percentage of occupied housing units that are renter occupied."
        ),
        direction="contextual",
        intervention=(
            "Housing-stability support"
        ),
    ),

    "housing_crowded_1_01_to_1_50_pct": SDOHFactor(
        key="housing_crowded_1_01_to_1_50_pct",
        name="Moderately crowded housing",
        domain_key="housing",
        description=(
            "Percentage of housing units with moderate crowding."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Housing-stability support"
        ),
    ),

    "housing_crowded_1_51_plus_pct": SDOHFactor(
        key="housing_crowded_1_51_plus_pct",
        name="Severely crowded housing",
        domain_key="housing",
        description=(
            "Percentage of housing units with severe crowding."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Housing-stability support"
        ),
    ),

    "housing_rent_30_to_34_9_pct": SDOHFactor(
        key="housing_rent_30_to_34_9_pct",
        name="Moderate housing cost burden",
        domain_key="housing",
        description=(
            "Percentage of renter households spending a moderate "
            "share of income on rent."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Housing and financial assistance"
        ),
    ),

    "housing_rent_35_plus_pct": SDOHFactor(
        key="housing_rent_35_plus_pct",
        name="Severe housing cost burden",
        domain_key="housing",
        description=(
            "Percentage of renter households spending a high share "
            "of income on rent."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Housing and financial assistance"
        ),
    ),

    # ================================================================
    # TRANSPORTATION
    # ================================================================

    "households_without_vehicle_count_sum": SDOHFactor(
        key="households_without_vehicle_count_sum",
        name="Households without vehicle",
        domain_key="transportation",
        description=(
            "Number of households without access to a vehicle."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation assistance for healthcare access"
        ),
    ),

    "driving_low_access_population_beyond_1mi_10mi_count_sum": SDOHFactor(
        key="driving_low_access_population_beyond_1mi_10mi_count_sum",
        name="Low-access population",
        domain_key="transportation",
        description=(
            "Population experiencing low geographic access "
            "to essential destinations."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation assistance for healthcare access"
        ),
    ),

    "driving_no_vehicle_households_beyond_1mi_count_sum": SDOHFactor(
        key="driving_no_vehicle_households_beyond_1mi_count_sum",
        name="No-vehicle households beyond access threshold",
        domain_key="transportation",
        description=(
            "Households without vehicles located beyond the "
            "defined access threshold."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation assistance for healthcare access"
        ),
    ),

    "driving_snap_households_beyond_1mi_count_sum": SDOHFactor(
        key="driving_snap_households_beyond_1mi_count_sum",
        name="SNAP households with transportation barriers",
        domain_key="transportation",
        description=(
            "SNAP households experiencing geographic transportation "
            "access constraints."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation and benefits coordination"
        ),
    ),

    "straight_no_vehicle_households_beyond_1mi_count_sum": SDOHFactor(
        key="straight_no_vehicle_households_beyond_1mi_count_sum",
        name="No-vehicle households with straight-line access barrier",
        domain_key="transportation",
        description=(
            "No-vehicle households experiencing straight-line "
            "geographic access constraints."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation assistance for healthcare access"
        ),
    ),

    "straight_snap_households_beyond_1mi_count_sum": SDOHFactor(
        key="straight_snap_households_beyond_1mi_count_sum",
        name="SNAP households with straight-line access barrier",
        domain_key="transportation",
        description=(
            "SNAP households experiencing straight-line "
            "geographic access constraints."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation and benefits coordination"
        ),
    ),

    "driving_low_income_low_access_tract_count": SDOHFactor(
        key="driving_low_income_low_access_tract_count",
        name="Low-income low-access tracts",
        domain_key="transportation",
        description=(
            "Number of census tracts characterized by low income "
            "and low geographic access."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation assistance for healthcare access"
        ),
    ),

    "driving_low_vehicle_access_tract_count": SDOHFactor(
        key="driving_low_vehicle_access_tract_count",
        name="Low vehicle-access tracts",
        domain_key="transportation",
        description=(
            "Number of census tracts with low vehicle access."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation assistance for healthcare access"
        ),
    ),

    "straight_low_income_low_access_tract_count": SDOHFactor(
        key="straight_low_income_low_access_tract_count",
        name="Straight-line low-income low-access tracts",
        domain_key="transportation",
        description=(
            "Number of tracts characterized by low income and "
            "low straight-line geographic access."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation assistance for healthcare access"
        ),
    ),

    "straight_low_vehicle_access_tract_count": SDOHFactor(
        key="straight_low_vehicle_access_tract_count",
        name="Straight-line low vehicle-access tracts",
        domain_key="transportation",
        description=(
            "Number of tracts with low straight-line vehicle access."
        ),
        direction="higher_is_greater_need",
        intervention=(
            "Transportation assistance for healthcare access"
        ),
    ),

    # ================================================================
    # DIGITAL ACCESS
    # ================================================================

    "digital_with_computer_pct": SDOHFactor(
        key="digital_with_computer_pct",
        name="Computer access",
        domain_key="digital_access",
        description=(
            "Percentage of households with access to a computer."
        ),
        direction="lower_is_greater_need",
        intervention=(
            "Digital-access and telehealth support"
        ),
    ),

    "digital_with_broadband_pct": SDOHFactor(
        key="digital_with_broadband_pct",
        name="Broadband access",
        domain_key="digital_access",
        description=(
            "Percentage of households with broadband internet access."
        ),
        direction="lower_is_greater_need",
        intervention=(
            "Digital-access and telehealth support"
        ),
    ),

    # ================================================================
    # NEIGHBORHOOD / BUILT ENVIRONMENT
    # ================================================================

    "places_asthma_pct": SDOHFactor(
        key="places_asthma_pct",
        name="Asthma prevalence",
        domain_key="neighborhood_built_environment",
        description="County-level asthma prevalence.",
        direction="higher_is_greater_need",
        intervention="Community health support",
    ),

    "places_copd_pct": SDOHFactor(
        key="places_copd_pct",
        name="COPD prevalence",
        domain_key="neighborhood_built_environment",
        description="County-level COPD prevalence.",
        direction="higher_is_greater_need",
        intervention="Chronic-condition support",
    ),

    "places_diabetes_pct": SDOHFactor(
        key="places_diabetes_pct",
        name="Diabetes prevalence",
        domain_key="neighborhood_built_environment",
        description="County-level diabetes prevalence.",
        direction="higher_is_greater_need",
        intervention="Chronic-condition support",
    ),

    "places_heart_disease_pct": SDOHFactor(
        key="places_heart_disease_pct",
        name="Heart disease prevalence",
        domain_key="neighborhood_built_environment",
        description="County-level heart disease prevalence.",
        direction="higher_is_greater_need",
        intervention="Chronic-condition support",
    ),

    "places_obesity_pct": SDOHFactor(
        key="places_obesity_pct",
        name="Obesity prevalence",
        domain_key="neighborhood_built_environment",
        description="County-level obesity prevalence.",
        direction="higher_is_greater_need",
        intervention="Lifestyle and preventive-health support",
    ),

    "places_physical_inactivity_pct": SDOHFactor(
        key="places_physical_inactivity_pct",
        name="Physical inactivity prevalence",
        domain_key="neighborhood_built_environment",
        description="County-level physical inactivity prevalence.",
        direction="higher_is_greater_need",
        intervention="Lifestyle and preventive-health support",
    ),

    "places_poor_mental_health_pct": SDOHFactor(
        key="places_poor_mental_health_pct",
        name="Poor mental health prevalence",
        domain_key="neighborhood_built_environment",
        description="County-level poor mental health prevalence.",
        direction="higher_is_greater_need",
        intervention="Behavioral-health support",
    ),

    "places_poor_physical_health_pct": SDOHFactor(
        key="places_poor_physical_health_pct",
        name="Poor physical health prevalence",
        domain_key="neighborhood_built_environment",
        description="County-level poor physical health prevalence.",
        direction="higher_is_greater_need",
        intervention="Community health support",
    ),

    "places_smoking_pct": SDOHFactor(
        key="places_smoking_pct",
        name="Smoking prevalence",
        domain_key="neighborhood_built_environment",
        description="County-level smoking prevalence.",
        direction="higher_is_greater_need",
        intervention="Smoking-cessation support",
    ),

    "places_stroke_pct": SDOHFactor(
        key="places_stroke_pct",
        name="Stroke prevalence",
        domain_key="neighborhood_built_environment",
        description="County-level stroke prevalence.",
        direction="higher_is_greater_need",
        intervention="Chronic-condition support",
    ),
}


def get_sdoh_factor(key: str) -> SDOHFactor:
    """
    Return a factor by model-feature key.
    """

    normalized = str(key).strip()

    if normalized not in SDOH_FACTORS:
        raise KeyError(f"Unknown SDOH factor: {key}")

    return SDOH_FACTORS[normalized]


def get_all_sdoh_factors() -> List[SDOHFactor]:
    """
    Return all configured SDOH factors.
    """

    return list(SDOH_FACTORS.values())


def get_factors_by_domain(domain_key: str) -> List[SDOHFactor]:
    """
    Return all factors belonging to a domain.
    """

    normalized = str(domain_key).strip().lower()

    return [
        factor
        for factor in SDOH_FACTORS.values()
        if factor.domain_key == normalized
    ]


def validate_factors() -> None:
    """
    Validate the factor knowledge base.
    """

    if not SDOH_FACTORS:
        raise ValueError("No SDOH factors configured.")

    for key, factor in SDOH_FACTORS.items():

        if factor.key != key:
            raise ValueError(
                f"Factor key mismatch: {key} != {factor.key}"
            )

        if not factor.name.strip():
            raise ValueError(
                f"Factor {key} has no name."
            )

        if not factor.domain_key.strip():
            raise ValueError(
                f"Factor {key} has no domain."
            )

        if factor.direction not in {
            "higher_is_greater_need",
            "lower_is_greater_need",
            "contextual",
        }:
            raise ValueError(
                f"Invalid direction for factor {key}: "
                f"{factor.direction}"
            )

        if not factor.intervention.strip():
            raise ValueError(
                f"Factor {key} has no intervention."
            )


def _self_test() -> None:
    print("=" * 70)
    print("HEALTHLENS KNOWLEDGE BASE — SDOH FACTORS")
    print("=" * 70)

    validate_factors()

    print(
        f"Knowledge base version: 1.0.0"
    )
    print(
        f"SDOH factors:           {len(SDOH_FACTORS)}"
    )

    print()
    print("FACTORS BY DOMAIN")
    print("-" * 70)

    domains = sorted(
        {
            factor.domain_key
            for factor in SDOH_FACTORS.values()
        }
    )

    for domain in domains:

        factors = get_factors_by_domain(domain)

        print()
        print(
            f"{domain}: {len(factors)} factors"
        )

        for factor in factors:
            print(
                f"  - {factor.key}"
            )

    print()
    print("SDOH FACTOR SELF-TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    _self_test()