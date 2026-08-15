from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member_sdoh_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "member_sdoh_model_features.csv"
)

CATALOG_FILE = (
    OUTPUT_DIR
    / "sdoh_feature_catalog.csv"
)


# Keys / identifiers retained for joining and traceability.
KEY_COLUMNS = [
    "member_id",
    "county_fips",
]


# Demographic/context variables kept separately from SDOH.
DEMOGRAPHIC_COLUMNS = [
    "age",
]


# Never use these directly as model features.
EXCLUDED_COLUMNS = {
    "zip",
    "fips",
    "lat",
    "lon",
    "state_fips",
    "places_year",

    # Previously identified redundant / structural variables.
    "population_2020_sum",
    "tract_count",
    "urban_tract_count",
    "straight_low_access_population_beyond_1mi_10mi_count_sum",
}


DOMAIN_MAP = {

    # Education
    "education_less_than_9th_pct": "education",
    "education_9th_to_12th_no_diploma_pct": "education",
    "education_high_school_pct": "education",
    "education_college_pct": "education",
    "education_bachelors_or_higher_pct": "education",

    # Economic stability
    "unemployment_pct": "economic_stability",
    "public_assistance_pct": "economic_stability",
    "poverty_pct": "economic_stability",
    "median_household_income": "economic_stability",
    "snap_households_count_sum": "economic_stability",

    # Healthcare access
    "uninsured_pct": "healthcare_access",
    "places_uninsured_pct": "healthcare_access",
    "places_routine_checkup_pct": "healthcare_access",
    "places_cholesterol_screening_pct": "healthcare_access",

    # Digital access
    "digital_with_computer_pct": "digital_access",
    "digital_with_broadband_pct": "digital_access",

    # Housing / built environment
    "housing_vacancy_pct": "neighborhood_built_environment",
    "housing_no_vehicle_pct": "neighborhood_built_environment",
    "housing_renter_pct": "neighborhood_built_environment",
    "housing_crowded_1_01_to_1_50_pct":
        "neighborhood_built_environment",
    "housing_crowded_1_51_plus_pct":
        "neighborhood_built_environment",
    "housing_rent_30_to_34_9_pct":
        "neighborhood_built_environment",
    "housing_rent_35_plus_pct":
        "neighborhood_built_environment",

    # Transportation
    "mean_commute_minutes": "transportation_access",
    "households_without_vehicle_count_sum":
        "transportation_access",
    "driving_low_access_population_beyond_1mi_10mi_count_sum":
        "transportation_access",
    "driving_no_vehicle_households_beyond_1mi_count_sum":
        "transportation_access",
    "driving_snap_households_beyond_1mi_count_sum":
        "transportation_access",
    "driving_low_income_low_access_tract_count":
        "transportation_access",
    "driving_low_vehicle_access_tract_count":
        "transportation_access",
    "straight_no_vehicle_households_beyond_1mi_count_sum":
        "transportation_access",
    "straight_snap_households_beyond_1mi_count_sum":
        "transportation_access",
    "straight_low_income_low_access_tract_count":
        "transportation_access",
    "straight_low_vehicle_access_tract_count":
        "transportation_access",

    # Health / behavioral context
    "places_asthma_pct": "health_behavioral_context",
    "places_copd_pct": "health_behavioral_context",
    "places_diabetes_pct": "health_behavioral_context",
    "places_heart_disease_pct":
        "health_behavioral_context",
    "places_obesity_pct": "health_behavioral_context",
    "places_physical_inactivity_pct":
        "health_behavioral_context",
    "places_poor_mental_health_pct":
        "health_behavioral_context",
    "places_poor_physical_health_pct":
        "health_behavioral_context",
    "places_smoking_pct": "health_behavioral_context",
    "places_stroke_pct": "health_behavioral_context",

    # Demographic context
    "population_total": "demographic_context",
    "median_age_years": "demographic_context",
}


def infer_source(feature):
    if feature.startswith("places_"):
        return "PLACES"

    if (
        feature.startswith("driving_")
        or feature.startswith("straight_")
        or feature.endswith("_tract_count")
        or feature.endswith("_count_sum")
    ):
        return "SRAM"

    return "ACS"


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------
    # Validate keys
    # --------------------------------------------------

    for column in KEY_COLUMNS:
        if column not in df.columns:
            raise ValueError(
                f"Missing required key column: {column}"
            )

    if df["member_id"].duplicated().any():
        raise ValueError(
            "member_id contains duplicates."
        )

    # --------------------------------------------------
    # Determine actual SDOH candidates
    # --------------------------------------------------

    available_domain_features = [
        feature
        for feature in DOMAIN_MAP
        if feature in df.columns
    ]

    if not available_domain_features:
        raise ValueError(
            "No configured SDOH features found."
        )

    # --------------------------------------------------
    # Explicit exclusion safety check
    # --------------------------------------------------

    accidental = (
        set(available_domain_features)
        & EXCLUDED_COLUMNS
    )

    if accidental:
        raise ValueError(
            "Excluded columns accidentally entered "
            f"the SDOH feature set: {sorted(accidental)}"
        )

    # --------------------------------------------------
    # Ensure all model features are numeric
    # --------------------------------------------------

    non_numeric = [
        feature
        for feature in available_domain_features
        if not pd.api.types.is_numeric_dtype(
            df[feature]
        )
    ]

    if non_numeric:
        raise ValueError(
            "Non-numeric SDOH features found: "
            f"{non_numeric}"
        )

    # --------------------------------------------------
    # Construct model dataset
    # --------------------------------------------------

    output_columns = (
        KEY_COLUMNS
        + DEMOGRAPHIC_COLUMNS
        + available_domain_features
    )

    output = df[output_columns].copy()

    # --------------------------------------------------
    # Build feature catalog
    # --------------------------------------------------

    catalog_rows = []

    for feature in available_domain_features:

        catalog_rows.append(
            {
                "feature": feature,
                "domain": DOMAIN_MAP[feature],
                "source": infer_source(feature),
                "included_in_model": True,
                "exclusion_reason": "",
            }
        )

    for feature in sorted(EXCLUDED_COLUMNS):

        if feature in df.columns:

            catalog_rows.append(
                {
                    "feature": feature,
                    "domain": "",
                    "source": infer_source(feature),
                    "included_in_model": False,
                    "exclusion_reason":
                        "Identifier, geography, metadata, "
                        "or redundant structural feature",
                }
            )

    for feature in DEMOGRAPHIC_COLUMNS:

        if feature in df.columns:

            catalog_rows.append(
                {
                    "feature": feature,
                    "domain": "demographic_context",
                    "source": "Synthea",
                    "included_in_model": False,
                    "exclusion_reason":
                        "Demographic covariate retained "
                        "separately from SDOH features",
                }
            )

    catalog = pd.DataFrame(catalog_rows)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    catalog.to_csv(
        CATALOG_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print("=" * 60)
    print("SDOH MODEL FEATURE SELECTION")
    print("=" * 60)

    print(f"Members: {len(output)}")
    print(
        f"SDOH model features: "
        f"{len(available_domain_features)}"
    )

    print(
        f"Demographic covariates retained separately: "
        f"{len([x for x in DEMOGRAPHIC_COLUMNS if x in df.columns])}"
    )

    print("\nSDOH features:")

    for feature in available_domain_features:
        print(
            f"  [{DOMAIN_MAP[feature]}] "
            f"{feature}"
        )

    print("\nMissing SDOH values:")

    missing = (
        output[available_domain_features]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    for feature, count in missing.items():

        if count > 0:

            pct = (
                count
                / len(output)
                * 100
            )

            print(
                f"  {feature}: "
                f"{count} "
                f"({pct:.2f}%)"
            )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nCatalog:")
    print(CATALOG_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()