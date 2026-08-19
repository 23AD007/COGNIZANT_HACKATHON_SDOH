from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"


# ============================================================
# INPUT DATASETS
# ============================================================

# SDOH-only member-level features
FEATURE_FILE = (
    PROCESSED_DIR / "member_sdoh_model_features.csv"
)

# Clinical target / outcome labels
TARGET_FILE = (
    PROCESSED_DIR / "member_clinical_target.csv"
)

# Final merged member-level modeling dataset
# Created by merge_member_features.py
MEMBER_MODEL_DATASET_FILE = (
    PROCESSED_DIR / "member_model_features.csv"
)


# ============================================================
# MODEL OUTPUTS
# ============================================================

MODEL_COMPARISON_FILE = (
    PROCESSED_DIR / "model_comparison_cv.csv"
)

FOLD_RESULTS_FILE = (
    PROCESSED_DIR / "model_comparison_fold_results.csv"
)

OOF_FILE = (
    PROCESSED_DIR / "model_oof_predictions.csv"
)

MEMBER_RISK_FILE = (
    PROCESSED_DIR / "member_risk_scores.csv"
)

COUNTY_RISK_FILE = (
    PROCESSED_DIR / "county_risk_scores.csv"
)

INTERVENTION_FILE = (
    PROCESSED_DIR / "intervention_priorities.csv"
)


# ============================================================
# TARGET
# ============================================================

TARGET_COLUMN = "target_inpatient_any"


# ============================================================
# IDENTIFIERS
# ============================================================

ID_COLUMNS = {
    "member_id",
    "patient_id",
}


# ============================================================
# GEOGRAPHY
# ============================================================

GEOGRAPHY_COLUMNS = {
    "county_fips",
    "zip",
    "fips",
    "lat",
    "lon",
}


# ============================================================
# TARGET / LEAKAGE COLUMNS
# ============================================================

TARGET_LEAKAGE_COLUMNS = {
    "inpatient_count",
    "encounter_count",
    "emergency_count",
    "urgent_care_count",
    "target_emergency_any",
    "target_inpatient_any",
    "target_acute_any",
    "target_acute_2plus",
    "target_top25_utilization",
    "selected_target",
    "target_definition",
}


# ============================================================
# MODELING CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5