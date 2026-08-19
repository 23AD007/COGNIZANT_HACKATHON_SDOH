from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

MEMBER_RISK_FILE = PROCESSED_DIR / "member_risk_scores.csv"
MEMBER_FEATURE_FILE = PROCESSED_DIR / "member_model_features.csv"
COUNTY_RISK_FILE = PROCESSED_DIR / "county_risk_scores.csv"

OUTPUT_FILE = PROCESSED_DIR / "intervention_priorities.csv"
VALIDATION_FILE = PROCESSED_DIR / "intervention_priorities_validation.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

HIGH_RISK_BANDS = {
    "High",
    "Very High",
}

VERY_HIGH_RISK_BAND = "Very High"


# ============================================================
# SDOH FACTOR DEFINITIONS
# ============================================================

# Each factor:
#
#   column       -> feature in member_model_features.csv
#   domain       -> SDOH domain
#   direction    -> whether a higher value indicates greater need
#   label        -> human-readable factor
#   intervention -> suggested intervention
#
# percentile-based scoring is used later, so this works even
# when the raw variables have different units.

SDOH_FACTORS: List[Dict[str, str]] = [

    # --------------------------------------------------------
    # ECONOMIC STABILITY
    # --------------------------------------------------------

    {
        "column": "poverty_pct",
        "domain": "Economic Stability",
        "direction": "high",
        "label": "High poverty",
        "intervention": "Financial assistance and benefits navigation",
    },
    {
        "column": "unemployment_pct",
        "domain": "Economic Stability",
        "direction": "high",
        "label": "High unemployment",
        "intervention": "Employment and economic-support referral",
    },
    {
        "column": "public_assistance_pct",
        "domain": "Economic Stability",
        "direction": "high",
        "label": "High public-assistance reliance",
        "intervention": "Benefits enrollment and financial-support navigation",
    },
    {
        "column": "median_household_income",
        "domain": "Economic Stability",
        "direction": "low",
        "label": "Low median household income",
        "intervention": "Financial assistance and benefits navigation",
    },
    {
        "column": "snap_households_count_sum",
        "domain": "Economic Stability",
        "direction": "high",
        "label": "High SNAP household concentration",
        "intervention": "Food-security and SNAP support",
    },

    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    {
        "column": "education_less_than_9th_pct",
        "domain": "Education Access",
        "direction": "high",
        "label": "Low educational attainment",
        "intervention": "Health-literacy and education support",
    },
    {
        "column": "education_9th_to_12th_no_diploma_pct",
        "domain": "Education Access",
        "direction": "high",
        "label": "High proportion without high-school diploma",
        "intervention": "Education and health-literacy support",
    },

    # --------------------------------------------------------
    # HEALTHCARE ACCESS
    # --------------------------------------------------------

    {
        "column": "uninsured_pct",
        "domain": "Healthcare Access",
        "direction": "high",
        "label": "High uninsured population",
        "intervention": "Insurance enrollment and care-navigation support",
    },
    {
        "column": "places_uninsured_pct",
        "domain": "Healthcare Access",
        "direction": "high",
        "label": "High uninsured prevalence",
        "intervention": "Insurance enrollment and care-navigation support",
    },
    {
        "column": "places_routine_checkup_pct",
        "domain": "Healthcare Access",
        "direction": "low",
        "label": "Low routine-checkup utilization",
        "intervention": "Primary-care outreach and appointment navigation",
    },
    {
        "column": "places_cholesterol_screening_pct",
        "domain": "Healthcare Access",
        "direction": "low",
        "label": "Low preventive screening",
        "intervention": "Preventive-screening outreach",
    },

    # --------------------------------------------------------
    # DIGITAL ACCESS
    # --------------------------------------------------------

    {
        "column": "digital_with_computer_pct",
        "domain": "Digital Access",
        "direction": "low",
        "label": "Low computer access",
        "intervention": "Digital-access and technology support",
    },
    {
        "column": "digital_with_broadband_pct",
        "domain": "Digital Access",
        "direction": "low",
        "label": "Low broadband access",
        "intervention": "Broadband and digital-health access support",
    },

    # --------------------------------------------------------
    # HOUSING
    # --------------------------------------------------------

    {
        "column": "housing_vacancy_pct",
        "domain": "Housing",
        "direction": "high",
        "label": "High housing vacancy",
        "intervention": "Housing-stability assessment",
    },
    {
        "column": "housing_no_vehicle_pct",
        "domain": "Housing",
        "direction": "high",
        "label": "High households without vehicles",
        "intervention": "Housing and transportation-support assessment",
    },
    {
        "column": "housing_renter_pct",
        "domain": "Housing",
        "direction": "high",
        "label": "High renter concentration",
        "intervention": "Housing-stability and tenant-support referral",
    },
    {
        "column": "housing_crowded_1_01_to_1_50_pct",
        "domain": "Housing",
        "direction": "high",
        "label": "Moderate household crowding",
        "intervention": "Housing-stability assessment",
    },
    {
        "column": "housing_crowded_1_51_plus_pct",
        "domain": "Housing",
        "direction": "high",
        "label": "Severe household crowding",
        "intervention": "Housing-stability and overcrowding support",
    },
    {
        "column": "housing_rent_30_to_34_9_pct",
        "domain": "Housing",
        "direction": "high",
        "label": "High housing cost burden",
        "intervention": "Housing-cost and rental assistance",
    },
    {
        "column": "housing_rent_35_plus_pct",
        "domain": "Housing",
        "direction": "high",
        "label": "Severe housing cost burden",
        "intervention": "Housing-cost and rental assistance",
    },

    # --------------------------------------------------------
    # TRANSPORTATION
    # --------------------------------------------------------

    {
        "column": "households_without_vehicle_count_sum",
        "domain": "Transportation",
        "direction": "high",
        "label": "High households without vehicles",
        "intervention": "Transportation assistance for healthcare access",
    },
    {
        "column": "driving_low_access_population_beyond_1mi_10mi_count_sum",
        "domain": "Transportation",
        "direction": "high",
        "label": "Low-access population",
        "intervention": "Transportation and care-access assistance",
    },
    {
        "column": "driving_no_vehicle_households_beyond_1mi_count_sum",
        "domain": "Transportation",
        "direction": "high",
        "label": "No-vehicle households with low food access",
        "intervention": "Transportation and food-access support",
    },
    {
        "column": "driving_snap_households_beyond_1mi_count_sum",
        "domain": "Transportation",
        "direction": "high",
        "label": "SNAP households with transportation barriers",
        "intervention": "Transportation and food-access support",
    },
    {
        "column": "driving_low_income_low_access_tract_count",
        "domain": "Transportation",
        "direction": "high",
        "label": "Low-income low-access areas",
        "intervention": "Transportation and community resource navigation",
    },
    {
        "column": "driving_low_vehicle_access_tract_count",
        "domain": "Transportation",
        "direction": "high",
        "label": "Low vehicle-access areas",
        "intervention": "Transportation assistance for healthcare access",
    },
    {
        "column": "straight_no_vehicle_households_beyond_1mi_count_sum",
        "domain": "Transportation",
        "direction": "high",
        "label": "No-vehicle households beyond access threshold",
        "intervention": "Transportation assistance",
    },
    {
        "column": "straight_snap_households_beyond_1mi_count_sum",
        "domain": "Transportation",
        "direction": "high",
        "label": "SNAP households beyond access threshold",
        "intervention": "Transportation and food-access support",
    },
    {
        "column": "straight_low_income_low_access_tract_count",
        "domain": "Transportation",
        "direction": "high",
        "label": "Low-income low-access tracts",
        "intervention": "Transportation and community resource navigation",
    },
    {
        "column": "straight_low_vehicle_access_tract_count",
        "domain": "Transportation",
        "direction": "high",
        "label": "Low vehicle-access tracts",
        "intervention": "Transportation assistance",
    },

    # --------------------------------------------------------
    # HEALTH / BUILT ENVIRONMENT
    # --------------------------------------------------------

    {
        "column": "places_asthma_pct",
        "domain": "Health Environment",
        "direction": "high",
        "label": "High asthma prevalence",
        "intervention": "Chronic respiratory-disease support",
    },
    {
        "column": "places_copd_pct",
        "domain": "Health Environment",
        "direction": "high",
        "label": "High COPD prevalence",
        "intervention": "Respiratory-disease management support",
    },
    {
        "column": "places_diabetes_pct",
        "domain": "Health Environment",
        "direction": "high",
        "label": "High diabetes prevalence",
        "intervention": "Diabetes prevention and care management",
    },
    {
        "column": "places_heart_disease_pct",
        "domain": "Health Environment",
        "direction": "high",
        "label": "High heart-disease prevalence",
        "intervention": "Cardiovascular-risk management",
    },
    {
        "column": "places_obesity_pct",
        "domain": "Health Environment",
        "direction": "high",
        "label": "High obesity prevalence",
        "intervention": "Nutrition and healthy-lifestyle support",
    },
    {
        "column": "places_physical_inactivity_pct",
        "domain": "Health Environment",
        "direction": "high",
        "label": "High physical inactivity",
        "intervention": "Physical-activity and wellness support",
    },
    {
        "column": "places_poor_mental_health_pct",
        "domain": "Health Environment",
        "direction": "high",
        "label": "Poor mental-health prevalence",
        "intervention": "Behavioral-health screening and referral",
    },
    {
        "column": "places_poor_physical_health_pct",
        "domain": "Health Environment",
        "direction": "high",
        "label": "Poor physical-health prevalence",
        "intervention": "Primary-care and chronic-condition support",
    },
    {
        "column": "places_smoking_pct",
        "domain": "Health Environment",
        "direction": "high",
        "label": "High smoking prevalence",
        "intervention": "Smoking-cessation support",
    },
    {
        "column": "places_stroke_pct",
        "domain": "Health Environment",
        "direction": "high",
        "label": "High stroke prevalence",
        "intervention": "Cardiovascular and stroke-risk management",
    },
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def normalize_member_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
    )


def normalize_county(series: pd.Series) -> pd.Series:
    """
    Normalize county FIPS without inventing missing geography.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    result = numeric.round().astype("Int64").astype("string")

    result = result.where(
        result.notna() & (result != "<NA>"),
        pd.NA,
    )

    return result


# ============================================================
# LOADING
# ============================================================

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    section("LOADING INTERVENTION INPUTS")

    if not MEMBER_RISK_FILE.exists():
        raise FileNotFoundError(
            f"Member risk file not found:\n{MEMBER_RISK_FILE}"
        )

    if not MEMBER_FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Member feature file not found:\n{MEMBER_FEATURE_FILE}"
        )

    if not COUNTY_RISK_FILE.exists():
        raise FileNotFoundError(
            f"County risk file not found:\n{COUNTY_RISK_FILE}"
        )

    risk = pd.read_csv(MEMBER_RISK_FILE)
    features = pd.read_csv(MEMBER_FEATURE_FILE)
    county = pd.read_csv(COUNTY_RISK_FILE)

    print(f"Member risk rows:     {len(risk)}")
    print(f"Member feature rows:  {len(features)}")
    print(f"County risk rows:     {len(county)}")

    return risk, features, county


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs(
    risk: pd.DataFrame,
    features: pd.DataFrame,
    county: pd.DataFrame,
) -> None:

    section("VALIDATING INPUTS")

    required_risk = {
        "member_id",
        "risk_probability",
        "risk_percentile",
        "risk_band",
    }

    required_features = {
        "member_id",
    }

    required_county = {
        "county_fips",
    }

    missing = required_risk - set(risk.columns)

    if missing:
        raise ValueError(
            f"Member risk missing required columns: {sorted(missing)}"
        )

    missing = required_features - set(features.columns)

    if missing:
        raise ValueError(
            f"Member features missing required columns: {sorted(missing)}"
        )

    missing = required_county - set(county.columns)

    if missing:
        raise ValueError(
            f"County risk missing required columns: {sorted(missing)}"
        )

    risk["member_id"] = normalize_member_id(risk["member_id"])
    features["member_id"] = normalize_member_id(features["member_id"])

    if risk["member_id"].duplicated().any():
        raise ValueError("Duplicate member_id in member risk data.")

    if features["member_id"].duplicated().any():
        raise ValueError("Duplicate member_id in member features.")

    risk["risk_probability"] = pd.to_numeric(
        risk["risk_probability"],
        errors="coerce",
    )

    if risk["risk_probability"].isna().any():
        raise ValueError(
            "Member risk contains missing risk_probability values."
        )

    if not risk["risk_probability"].between(0, 1).all():
        raise ValueError(
            "risk_probability must be between 0 and 1."
        )

    print(f"Risk members:       {len(risk)}")
    print(f"Feature members:    {len(features)}")
    print(
        f"High-risk members:  "
        f"{risk['risk_band'].isin(HIGH_RISK_BANDS).sum()}"
    )

    print("Input validation: PASSED")


# ============================================================
# PREPARE FEATURE DATA
# ============================================================

def prepare_feature_data(
    features: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:

    section("PREPARING SDOH FACTORS")

    available = []

    for factor in SDOH_FACTORS:
        column = factor["column"]

        if column in features.columns:
            available.append(factor)

    print(f"Configured SDOH factors: {len(SDOH_FACTORS)}")
    print(f"Available SDOH factors:  {len(available)}")

    if not available:
        raise ValueError(
            "None of the configured SDOH factors exist in "
            "member_model_features.csv."
        )

    factor_metadata = {}

    for factor in available:

        column = factor["column"]

        values = pd.to_numeric(
            features[column],
            errors="coerce",
        )

        features[column] = values

        factor_metadata[column] = factor

    return features, factor_metadata


# ============================================================
# PERCENTILE / NEED SCORING
# ============================================================

def calculate_factor_scores(
    features: pd.DataFrame,
    factor_metadata: Dict[str, Dict[str, float]],
) -> pd.DataFrame:

    section("CALCULATING SDOH NEED SCORES")

    result = pd.DataFrame(index=features.index)

    for column, metadata in factor_metadata.items():

        values = pd.to_numeric(
            features[column],
            errors="coerce",
        )

        # Percentile rank among available members.
        percentile = values.rank(
            pct=True,
            method="average",
        )

        # Higher percentile means greater need.
        if metadata["direction"] == "high":
            score = percentile

        elif metadata["direction"] == "low":
            score = 1.0 - percentile

        else:
            raise ValueError(
                f"Unknown direction for {column}: "
                f"{metadata['direction']}"
            )

        result[column] = score

    return result


# ============================================================
# DOMAIN SCORING
# ============================================================

def calculate_domain_scores(
    factor_scores: pd.DataFrame,
    factor_metadata: Dict[str, Dict[str, float]],
) -> pd.DataFrame:

    section("CALCULATING SDOH DOMAIN SCORES")

    domains = sorted(
        {
            metadata["domain"]
            for metadata in factor_metadata.values()
        }
    )

    domain_scores = pd.DataFrame(
        index=factor_scores.index
    )

    for domain in domains:

        columns = [
            column
            for column, metadata in factor_metadata.items()
            if metadata["domain"] == domain
        ]

        if not columns:
            continue

        # Mean of available factor scores.
        domain_scores[domain] = factor_scores[
            columns
        ].mean(axis=1, skipna=True)

    return domain_scores


# ============================================================
# TOP FACTOR
# ============================================================

def get_top_factor(
    row: pd.Series,
    factor_scores: pd.DataFrame,
    factor_metadata: Dict[str, Dict[str, float]],
) -> Tuple[str, str, str, float]:

    available_scores = row.dropna()

    if available_scores.empty:
        return (
            "Unknown",
            "No measurable SDOH factor",
            "No intervention recommendation",
            np.nan,
        )

    top_column = available_scores.idxmax()

    metadata = factor_metadata[top_column]

    return (
        metadata["domain"],
        metadata["label"],
        metadata["intervention"],
        float(available_scores[top_column]),
    )


# ============================================================
# DOMAIN INTERVENTION
# ============================================================

DOMAIN_INTERVENTIONS = {
    "Economic Stability":
        "Financial assistance and benefits navigation",

    "Education Access":
        "Health-literacy and education support",

    "Healthcare Access":
        "Primary-care, insurance, and care-navigation support",

    "Digital Access":
        "Digital-access and telehealth support",

    "Housing":
        "Housing-stability and housing-cost support",

    "Transportation":
        "Transportation assistance for healthcare access",

    "Health Environment":
        "Preventive and chronic-condition support",
}


def domain_intervention(domain: str) -> str:

    return DOMAIN_INTERVENTIONS.get(
        domain,
        "Community resource navigation",
    )


# ============================================================
# PRIORITY SCORE
# ============================================================

def calculate_priority_score(
    risk_probability: float,
    sdoh_score: float,
) -> float:

    if pd.isna(risk_probability):
        return np.nan

    if pd.isna(sdoh_score):
        sdoh_score = 0.0

    # Risk receives slightly greater weight than SDOH need.
    score = (
        0.60 * float(risk_probability)
        + 0.40 * float(sdoh_score)
    )

    return float(score)


# ============================================================
# BUILD MEMBER PRIORITIES
# ============================================================

def build_member_priorities(
    risk: pd.DataFrame,
    features: pd.DataFrame,
    factor_scores: pd.DataFrame,
    domain_scores: pd.DataFrame,
    factor_metadata: Dict[str, Dict[str, float]],
) -> pd.DataFrame:

    section("BUILDING MEMBER INTERVENTION PRIORITIES")

    df = risk.merge(
        features,
        on="member_id",
        how="left",
        suffixes=("", "_feature"),
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Attach factor scores to the merged frame.
    # --------------------------------------------------------

    score_columns = {}

    for column in factor_scores.columns:

        output_column = f"sdoh_factor_score__{column}"

        df[output_column] = factor_scores[column].values

        score_columns[column] = output_column

    # --------------------------------------------------------
    # Domain scores.
    # --------------------------------------------------------

    domain_output_columns = {}

    for domain in domain_scores.columns:

        safe_name = (
            domain.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        output_column = f"sdoh_domain_score__{safe_name}"

        df[output_column] = domain_scores[domain].values

        domain_output_columns[domain] = output_column

    # --------------------------------------------------------
    # Primary and secondary domain.
    # --------------------------------------------------------

    primary_domains = []
    secondary_domains = []
    primary_domain_scores = []
    secondary_domain_scores = []

    for idx in range(len(df)):

        row = domain_scores.iloc[idx].dropna()

        if row.empty:

            primary_domains.append("Unknown")
            secondary_domains.append("Unknown")
            primary_domain_scores.append(np.nan)
            secondary_domain_scores.append(np.nan)

            continue

        ranked = row.sort_values(
            ascending=False
        )

        primary_domains.append(ranked.index[0])
        primary_domain_scores.append(float(ranked.iloc[0]))

        if len(ranked) > 1:
            secondary_domains.append(ranked.index[1])
            secondary_domain_scores.append(float(ranked.iloc[1]))
        else:
            secondary_domains.append("Unknown")
            secondary_domain_scores.append(np.nan)

    df["primary_sdoh_domain"] = primary_domains
    df["secondary_sdoh_domain"] = secondary_domains
    df["primary_sdoh_score"] = primary_domain_scores
    df["secondary_sdoh_score"] = secondary_domain_scores

    # --------------------------------------------------------
    # Primary factor.
    # --------------------------------------------------------

    primary_factor = []
    primary_factor_label = []
    primary_intervention = []
    primary_factor_score = []

    for idx in range(len(df)):

        row = factor_scores.iloc[idx]

        (
            domain,
            label,
            intervention,
            score,
        ) = get_top_factor(
            row,
            factor_scores,
            factor_metadata,
        )

        primary_factor.append(domain)
        primary_factor_label.append(label)
        primary_intervention.append(intervention)
        primary_factor_score.append(score)

    df["primary_factor_domain"] = primary_factor
    df["primary_factor"] = primary_factor_label
    df["primary_factor_score"] = primary_factor_score

    # --------------------------------------------------------
    # Primary intervention based on dominant domain.
    # --------------------------------------------------------

    df["recommended_intervention"] = df[
        "primary_sdoh_domain"
    ].map(domain_intervention)

    # If a more specific factor recommendation exists,
    # use it.
    for i in range(len(df)):

        factor_domain = df.iloc[i]["primary_factor_domain"]

        if factor_domain in DOMAIN_INTERVENTIONS:
            specific = df.iloc[i]["primary_factor"]

            if pd.notna(specific):
                intervention = df.iloc[i][
                    "recommended_intervention"
                ]

                if pd.isna(intervention):
                    intervention = DOMAIN_INTERVENTIONS[
                        factor_domain
                    ]

                df.iat[
                    i,
                    df.columns.get_loc(
                        "recommended_intervention"
                    ),
                ] = intervention

    # --------------------------------------------------------
    # Overall SDOH score.
    # --------------------------------------------------------

    df["overall_sdoh_need_score"] = domain_scores.mean(
        axis=1,
        skipna=True,
    ).values

    # --------------------------------------------------------
    # Combined intervention priority.
    # --------------------------------------------------------

    df["intervention_priority_score"] = [
        calculate_priority_score(
            risk_probability,
            sdoh_score,
        )
        for risk_probability, sdoh_score
        in zip(
            df["risk_probability"],
            df["overall_sdoh_need_score"],
        )
    ]

    # --------------------------------------------------------
    # Priority classification.
    # --------------------------------------------------------

    df["intervention_priority"] = pd.cut(
        df["intervention_priority_score"],
        bins=[
            -np.inf,
            0.20,
            0.40,
            0.60,
            0.80,
            np.inf,
        ],
        labels=[
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
        include_lowest=True,
    ).astype("string")

    # --------------------------------------------------------
    # County.
    # --------------------------------------------------------

    if "county_fips" in df.columns:
        df["county_fips"] = normalize_county(
            df["county_fips"]
        )

    # --------------------------------------------------------
    # Rank.
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "intervention_priority_score",
            "risk_probability",
            "overall_sdoh_need_score",
        ],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    df["intervention_rank"] = np.arange(
        1,
        len(df) + 1,
    )

    # --------------------------------------------------------
    # High-risk action flag.
    # --------------------------------------------------------

    df["action_required"] = np.where(
        (
            df["risk_band"].isin(HIGH_RISK_BANDS)
            |
            (
                df["intervention_priority"].isin(
                    ["High", "Very High"]
                )
            )
        ),
        "Yes",
        "No",
    )

    return df


# ============================================================
# SELECT OUTPUT COLUMNS
# ============================================================

def select_output_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    columns = [
        "intervention_rank",
        "member_id",
        "patient_id",
        "county_fips",

        "risk_probability",
        "risk_percentile",
        "risk_band",

        "overall_sdoh_need_score",
        "intervention_priority_score",
        "intervention_priority",

        "primary_sdoh_domain",
        "primary_sdoh_score",

        "secondary_sdoh_domain",
        "secondary_sdoh_score",

        "primary_factor_domain",
        "primary_factor",
        "primary_factor_score",

        "recommended_intervention",

        "action_required",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    output = df[available].copy()

    return output


# ============================================================
# VALIDATION REPORT
# ============================================================

def create_validation_report(
    output: pd.DataFrame,
    factor_scores: pd.DataFrame,
    domain_scores: pd.DataFrame,
) -> pd.DataFrame:

    section("VALIDATING INTERVENTION OUTPUT")

    checks = []

    checks.append(
        {
            "check": "row_count",
            "value": len(output),
            "status": "PASS" if len(output) > 0 else "FAIL",
        }
    )

    checks.append(
        {
            "check": "unique_members",
            "value": output["member_id"].nunique(),
            "status": (
                "PASS"
                if output["member_id"].nunique() == len(output)
                else "FAIL"
            ),
        }
    )

    checks.append(
        {
            "check": "missing_risk_probability",
            "value": int(
                output["risk_probability"].isna().sum()
            ),
            "status": (
                "PASS"
                if not output["risk_probability"].isna().any()
                else "FAIL"
            ),
        }
    )

    checks.append(
        {
            "check": "risk_probability_out_of_range",
            "value": int(
                (
                    ~output["risk_probability"].between(0, 1)
                ).sum()
            ),
            "status": (
                "PASS"
                if output["risk_probability"].between(0, 1).all()
                else "FAIL"
            ),
        }
    )

    checks.append(
        {
            "check": "missing_primary_domain",
            "value": int(
                output["primary_sdoh_domain"]
                .isna()
                .sum()
            ),
            "status": "PASS",
        }
    )

    checks.append(
        {
            "check": "missing_intervention",
            "value": int(
                output["recommended_intervention"]
                .isna()
                .sum()
            ),
            "status": (
                "PASS"
                if not output[
                    "recommended_intervention"
                ].isna().any()
                else "FAIL"
            ),
        }
    )

    checks.append(
        {
            "check": "factor_score_columns",
            "value": factor_scores.shape[1],
            "status": (
                "PASS"
                if factor_scores.shape[1] > 0
                else "FAIL"
            ),
        }
    )

    checks.append(
        {
            "check": "domain_score_columns",
            "value": domain_scores.shape[1],
            "status": (
                "PASS"
                if domain_scores.shape[1] > 0
                else "FAIL"
            ),
        }
    )

    checks.append(
        {
            "check": "duplicate_member_ids",
            "value": int(
                output["member_id"].duplicated().sum()
            ),
            "status": (
                "PASS"
                if not output["member_id"].duplicated().any()
                else "FAIL"
            ),
        }
    )

    report = pd.DataFrame(checks)

    if (report["status"] == "FAIL").any():
        raise ValueError(
            "Intervention output validation failed."
        )

    return report


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    output: pd.DataFrame,
) -> None:

    section("INTERVENTION PRIORITIZATION SUMMARY")

    print(f"Members: {len(output)}")

    if "action_required" in output.columns:
        print(
            f"Action required: "
            f"{(output['action_required'] == 'Yes').sum()}"
        )

    print()

    print("Intervention priority:")
    print(
        output["intervention_priority"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print("Primary SDOH domains:")

    print(
        output["primary_sdoh_domain"]
        .value_counts()
        .to_string()
    )

    print()

    section("TOP 10 INTERVENTION PRIORITIES")

    display_columns = [
        "intervention_rank",
        "member_id",
        "risk_probability",
        "risk_band",
        "overall_sdoh_need_score",
        "intervention_priority_score",
        "primary_sdoh_domain",
        "primary_factor",
        "recommended_intervention",
    ]

    available = [
        c
        for c in display_columns
        if c in output.columns
    ]

    print(
        output[available]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    section("SDOH INTERVENTION PRIORITIZATION")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    risk, features, county = load_inputs()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_inputs(
        risk,
        features,
        county,
    )

    # --------------------------------------------------------
    # Prepare factors
    # --------------------------------------------------------

    features, factor_metadata = prepare_feature_data(
        features
    )

    # --------------------------------------------------------
    # Calculate factor need scores
    # --------------------------------------------------------

    factor_scores = calculate_factor_scores(
        features,
        factor_metadata,
    )

    # --------------------------------------------------------
    # Calculate domain scores
    # --------------------------------------------------------

    domain_scores = calculate_domain_scores(
        factor_scores,
        factor_metadata,
    )

    # --------------------------------------------------------
    # Build priorities
    # --------------------------------------------------------

    priorities = build_member_priorities(
        risk=risk,
        features=features,
        factor_scores=factor_scores,
        domain_scores=domain_scores,
        factor_metadata=factor_metadata,
    )

    # --------------------------------------------------------
    # Select clean output
    # --------------------------------------------------------

    output = select_output_columns(
        priorities
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation = create_validation_report(
        output,
        factor_scores,
        domain_scores,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(output)

    section("OUTPUT CREATED")

    print(f"Intervention priorities:")
    print(OUTPUT_FILE)

    print()

    print("Validation report:")
    print(VALIDATION_FILE)

    print()

    print("SDOH factors used:")
    print(len(factor_metadata))

    print("SDOH domains:")
    print(len(domain_scores.columns))

    print()

    print("=" * 70)
    print("INTERVENTION PRIORITIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()