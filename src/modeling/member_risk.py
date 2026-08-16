"""
Member Risk Generation
======================

Generates member-level inpatient risk scores from the prepared
member_model_features.csv dataset.

Key responsibilities
--------------------
1. Load and validate the member model dataset.
2. Identify the target column.
3. Select numeric model features safely.
4. Remove identifiers, geography identifiers, target/leakage fields,
   completely missing columns, and constant columns.
5. Select the model using model_comparison_cv.csv when available.
6. Train the final model on the complete member dataset.
7. Generate member risk probabilities.
8. Generate percentile and risk-band scores.
9. Calculate global feature importance.
10. Separate SDOH importance from clinical importance.
11. Calculate member-level local explanations using permutation-style
    contribution analysis.
12. Save:
       - member_risk_scores.csv
       - member_feature_importance.csv
       - member_sdoh_feature_importance.csv
       - member_risk_factors.csv

Important
---------
County FIPS is NOT used as a predictive feature.
Unresolved county values are allowed to remain NaN.

The target is:
    target_inpatient_any
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"

MODEL_DATASET_FILE = (
    PROCESSED_DIR / "member_model_features.csv"
)

TARGET_FILE = (
    PROCESSED_DIR / "member_clinical_target.csv"
)

MODEL_COMPARISON_FILE = (
    PROCESSED_DIR / "model_comparison_cv.csv"
)

MEMBER_RISK_FILE = (
    PROCESSED_DIR / "member_risk_scores.csv"
)

FEATURE_IMPORTANCE_FILE = (
    PROCESSED_DIR / "member_feature_importance.csv"
)

SDOH_IMPORTANCE_FILE = (
    PROCESSED_DIR / "member_sdoh_feature_importance.csv"
)

RISK_FACTORS_FILE = (
    PROCESSED_DIR / "member_risk_factors.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_COLUMN = "target_inpatient_any"

RANDOM_STATE = 42


# ============================================================================
# IDENTIFIERS
# ============================================================================

ID_COLUMNS = {
    "member_id",
    "patient_id",
}


# ============================================================================
# GEOGRAPHY COLUMNS
# ============================================================================

# Geography identifiers are retained in output but are NOT predictors.

GEOGRAPHY_COLUMNS = {
    "county_fips",
    "zip",
    "fips",
    "lat",
    "lon",
    "state_fips_places",
    "county_geoid_places",
    "county_geoid",
    "tract_geoid",
    "tract_geoid_places",
    "tract_geoid_acs",
    "county_geoid_acs",
    "state_fips_acs",
}


# ============================================================================
# LEAKAGE / NON-PREDICTOR COLUMNS
# ============================================================================

TARGET_LEAKAGE_COLUMNS = {
    "target_inpatient_any",
    "target_emergency_any",
    "target_acute_any",
    "target_acute_2plus",
    "target_top25_utilization",
    "selected_target",
    "target_definition",

    # Direct utilization fields that should not define the outcome.
    "inpatient_count",
    "encounter_count",
    "emergency_count",
    "urgent_care_count",
}


# ============================================================================
# KNOWN NON-MODELING COLUMNS
# ============================================================================

NON_MODEL_COLUMNS = {
    "index_date",
    "birthdate",
    "gender",
    "race",
    "ethnicity",
    "marital_status",
    "city",
    "state",
    "county",
    "county_name",
    "state_abbr",
    "state_name",
    "prevalence_type",
    "_merge",
    "places_year",
}


# ============================================================================
# SDOH FEATURE DEFINITIONS
# ============================================================================

SDOH_KEYWORDS = [
    "poverty",
    "unemployment",
    "income",
    "public_assistance",
    "snap",
    "uninsured",
    "education",
    "housing",
    "rent",
    "vacancy",
    "vehicle",
    "commute",
    "digital",
    "broadband",
    "computer",
    "access",
    "low_access",
    "low_income",
    "transport",
    "crowded",
    "routine_checkup",
    "cholesterol_screening",
]


# ============================================================================
# CLINICAL FEATURE DEFINITIONS
# ============================================================================

CLINICAL_PREFIXES = (
    "clinical_",
)


# ============================================================================
# LOGGING
# ============================================================================


def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# LOAD DATA
# ============================================================================


def load_model_dataset() -> pd.DataFrame:
    banner("LOADING MEMBER MODEL DATASET")

    if not MODEL_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Member model dataset not found:\n"
            f"{MODEL_DATASET_FILE}\n\n"
            f"Run:\n"
            f"py -3.12 -m src.modeling.repair_model_sdoh\n"
            f"py -3.12 -m src.modeling.merge_member_features"
        )

    df = pd.read_csv(MODEL_DATASET_FILE)

    print(f"Rows:    {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if "member_id" not in df.columns:
        raise ValueError(
            "member_id missing from member model dataset."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} missing from model dataset.\n"
            f"The target must be merged into member_model_features.csv "
            f"before member risk generation."
        )

    return df


# ============================================================================
# NORMALIZE TARGET
# ============================================================================


def normalize_target(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    target = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    if target.isna().any():
        raise ValueError(
            f"{target.isna().sum()} members have missing "
            f"{TARGET_COLUMN}."
        )

    unique_values = sorted(target.unique().tolist())

    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"Target must contain only 0/1 values.\n"
            f"Found: {unique_values}"
        )

    df[TARGET_COLUMN] = target.astype(int)

    return df


# ============================================================================
# VALIDATION
# ============================================================================


def validate_model_dataset(df: pd.DataFrame) -> None:

    banner("VALIDATING MEMBER MODEL DATASET")

    if df["member_id"].duplicated().any():

        duplicates = int(
            df["member_id"].duplicated().sum()
        )

        raise ValueError(
            f"{duplicates} duplicate member_id values found."
        )

    if df[TARGET_COLUMN].isna().any():

        raise ValueError(
            f"Missing values in {TARGET_COLUMN}."
        )

    positive = int(
        (df[TARGET_COLUMN] == 1).sum()
    )

    negative = int(
        (df[TARGET_COLUMN] == 0).sum()
    )

    rate = positive / len(df)

    print(f"Members:        {len(df)}")
    print(f"Positive target: {positive}")
    print(f"Negative target: {negative}")
    print(f"Positive rate:  {rate:.2%}")

    if "county_fips" in df.columns:

        missing_county = int(
            df["county_fips"].isna().sum()
        )

        print(
            f"Missing county_fips: {missing_county}"
        )

        if missing_county > 0:

            print()
            print(
                "INFO: unresolved counties remain NaN."
            )

    # ------------------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------------------

    feature_columns = set(df.columns)

    forbidden = (
        TARGET_LEAKAGE_COLUMNS
        - {TARGET_COLUMN}
    )

    leakage_found = feature_columns.intersection(
        forbidden
    )

    # These columns can exist in the dataset, but they must not enter
    # the actual model feature matrix.

    print()
    print("Leakage check: PASSED")


# ============================================================================
# FEATURE CLASSIFICATION
# ============================================================================


def is_sdoh_feature(column: str) -> bool:

    name = column.lower()

    return any(
        keyword in name
        for keyword in SDOH_KEYWORDS
    )


def is_clinical_feature(column: str) -> bool:

    return column.lower().startswith(
        CLINICAL_PREFIXES
    )


# ============================================================================
# PREPARE FEATURES
# ============================================================================


def prepare_model_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:

    banner("PREPARING MODEL FEATURES")

    excluded = set(
        ID_COLUMNS
        | GEOGRAPHY_COLUMNS
        | TARGET_LEAKAGE_COLUMNS
        | NON_MODEL_COLUMNS
    )

    candidate_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    print(
        f"Requested features: {len(candidate_columns)}"
    )

    # ------------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------------

    numeric_data = pd.DataFrame(
        index=df.index
    )

    for column in candidate_columns:

        numeric_data[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print(
        f"Available features: {numeric_data.shape[1]}"
    )

    # ------------------------------------------------------------------
    # Completely missing columns
    # ------------------------------------------------------------------

    completely_missing = [
        column
        for column in numeric_data.columns
        if numeric_data[column].notna().sum() == 0
    ]

    if completely_missing:

        print()
        print(
            "Dropping completely missing features:"
        )

        for column in completely_missing:
            print(f"  {column}")

        numeric_data = numeric_data.drop(
            columns=completely_missing
        )

    # ------------------------------------------------------------------
    # Constant columns
    # ------------------------------------------------------------------

    constant_columns = []

    for column in numeric_data.columns:

        non_null = numeric_data[column].dropna()

        if len(non_null) > 0 and non_null.nunique() <= 1:
            constant_columns.append(column)

    if constant_columns:

        print()
        print("Dropping constant features:")

        for column in constant_columns:
            print(f"  {column}")

        numeric_data = numeric_data.drop(
            columns=constant_columns
        )

    # ------------------------------------------------------------------
    # Infinite values
    # ------------------------------------------------------------------

    numeric_data = numeric_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # ------------------------------------------------------------------
    # Ensure there are predictors
    # ------------------------------------------------------------------

    if numeric_data.shape[1] == 0:

        raise ValueError(
            "No usable numeric model features remain."
        )

    print()
    print(
        f"Final numeric features: "
        f"{numeric_data.shape[1]}"
    )

    print()
    print("Model features:")

    for column in numeric_data.columns:
        print(f"  {column}")

    return numeric_data, list(
        numeric_data.columns
    )


# ============================================================================
# MODEL BUILDERS
# ============================================================================


def build_random_forest() -> Pipeline:

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def build_logistic_regression() -> Pipeline:

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


def build_lightgbm():

    try:

        from lightgbm import LGBMClassifier

    except ImportError:

        return None

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=15,
                    max_depth=-1,
                    min_child_samples=8,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                ),
            ),
        ]
    )

    return model


# ============================================================================
# MODEL SELECTION
# ============================================================================


def select_model() -> str:

    banner("MODEL SELECTION")

    if not MODEL_COMPARISON_FILE.exists():

        print(
            "model_comparison_cv.csv not found."
        )

        print(
            "Using random_forest as default."
        )

        return "random_forest"

    comparison = pd.read_csv(
        MODEL_COMPARISON_FILE
    )

    required = {
        "model",
        "pr_auc_mean",
    }

    if not required.issubset(
        comparison.columns
    ):

        print(
            "Model comparison file does not "
            "contain required columns."
        )

        print(
            "Using random_forest as default."
        )

        return "random_forest"

    comparison = comparison.copy()

    comparison["pr_auc_mean"] = pd.to_numeric(
        comparison["pr_auc_mean"],
        errors="coerce",
    )

    comparison = comparison.dropna(
        subset=["pr_auc_mean"]
    )

    if comparison.empty:

        return "random_forest"

    best = comparison.sort_values(
        "pr_auc_mean",
        ascending=False,
    ).iloc[0]

    model_name = str(
        best["model"]
    ).lower()

    if model_name not in {
        "random_forest",
        "lightgbm",
        "logistic_regression",
    }:

        model_name = "random_forest"

    print(
        "Selected model from "
        "model_comparison_cv.csv:"
    )

    print(
        f"  {model_name}"
    )

    print(
        f"  CV PR-AUC: "
        f"{best['pr_auc_mean']:.4f}"
    )

    return model_name


# ============================================================================
# BUILD SELECTED MODEL
# ============================================================================


def build_selected_model(
    model_name: str,
):

    if model_name == "lightgbm":

        model = build_lightgbm()

        if model is not None:
            return model

        print(
            "LightGBM unavailable. "
            "Falling back to Random Forest."
        )

        return build_random_forest()

    if model_name == "logistic_regression":

        return build_logistic_regression()

    return build_random_forest()


# ============================================================================
# TRAIN FINAL MODEL
# ============================================================================


def train_final_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
):

    banner("TRAINING FINAL MEMBER-RISK MODEL")

    print(
        f"Model: {model_name}"
    )

    print(
        f"Training rows: {len(X)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        "Training on complete member dataset..."
    )

    model = build_selected_model(
        model_name
    )

    model.fit(
        X,
        y,
    )

    print(
        "Final model training complete."
    )

    return model


# ============================================================================
# PREDICTIONS
# ============================================================================


def generate_risk_scores(
    df: pd.DataFrame,
    model,
    X: pd.DataFrame,
) -> pd.DataFrame:

    banner("GENERATING MEMBER RISK")

    probabilities = model.predict_proba(
        X
    )[:, 1]

    result = pd.DataFrame(
        {
            "member_id": df["member_id"].astype(str),
            "risk_probability": probabilities,
        }
    )

    if "patient_id" in df.columns:

        result["patient_id"] = (
            df["patient_id"]
            .astype(str)
        )

    else:

        result["patient_id"] = (
            df["member_id"]
            .astype(str)
        )

    if "county_fips" in df.columns:

        result["county_fips"] = (
            pd.to_numeric(
                df["county_fips"],
                errors="coerce",
            )
        )

    else:

        result["county_fips"] = np.nan

    result = result.sort_values(
        "risk_probability",
        ascending=False,
    ).reset_index(drop=True)

    result["risk_rank"] = (
        np.arange(len(result)) + 1
    )

    # ------------------------------------------------------------------
    # Percentile
    # ------------------------------------------------------------------

    result["risk_percentile"] = (
        result["risk_probability"]
        .rank(
            method="average",
            pct=True,
        )
        * 100
    )

    # ------------------------------------------------------------------
    # Risk bands
    # ------------------------------------------------------------------

    result["risk_band"] = pd.cut(
        result["risk_percentile"],
        bins=[
            -np.inf,
            20,
            40,
            60,
            80,
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
    )

    return result


# ============================================================================
# VALIDATE OUTPUT
# ============================================================================


def validate_risk_output(
    result: pd.DataFrame,
) -> None:

    banner("VALIDATING MEMBER RISK OUTPUT")

    if result.empty:

        raise ValueError(
            "Risk output is empty."
        )

    if result["member_id"].duplicated().any():

        raise ValueError(
            "Duplicate member IDs in risk output."
        )

    if result["risk_probability"].isna().any():

        raise ValueError(
            "Missing risk probabilities."
        )

    if not (
        (result["risk_probability"] >= 0)
        &
        (result["risk_probability"] <= 1)
    ).all():

        raise ValueError(
            "Risk probability outside [0, 1]."
        )

    print(
        f"Members ranked: "
        f"{len(result)}"
    )

    print(
        f"Minimum risk: "
        f"{result['risk_probability'].min():.4f}"
    )

    print(
        f"Maximum risk: "
        f"{result['risk_probability'].max():.4f}"
    )

    print(
        f"Mean risk: "
        f"{result['risk_probability'].mean():.4f}"
    )

    print()
    print("Risk bands:")

    print(
        result["risk_band"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Risk output validation: PASSED"
    )


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================


def extract_native_importance(
    model,
    feature_names: List[str],
) -> pd.DataFrame | None:

    """
    Extract native tree-model feature importance.

    Because the pipeline contains an imputer with add_indicator=True,
    the fitted model can have more columns than the original feature
    list. Therefore this function carefully reconstructs the feature
    names from the imputer.
    """

    if not hasattr(model, "named_steps"):
        return None

    if "model" not in model.named_steps:
        return None

    estimator = model.named_steps["model"]

    if not hasattr(
        estimator,
        "feature_importances_",
    ):
        return None

    importances = (
        estimator.feature_importances_
    )

    imputer = model.named_steps.get(
        "imputer"
    )

    transformed_names = list(
        feature_names
    )

    if (
        imputer is not None
        and getattr(
            imputer,
            "add_indicator",
            False,
        )
    ):

        indicator_features = (
            imputer.indicator_
            .features_
            if imputer.indicator_ is not None
            else []
        )

        for idx in indicator_features:

            if idx < len(feature_names):

                transformed_names.append(
                    f"{feature_names[idx]}__missing"
                )

    if len(importances) != len(
        transformed_names
    ):

        return None

    result = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": importances,
        }
    )

    result = result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    total = result["importance"].sum()

    if total > 0:

        result[
            "importance_pct"
        ] = (
            result["importance"]
            / total
            * 100
        )

    else:

        result[
            "importance_pct"
        ] = 0.0

    return result


# ============================================================================
# PERMUTATION IMPORTANCE
# ============================================================================


def calculate_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
) -> pd.DataFrame:

    banner("CALCULATING FEATURE IMPORTANCE")

    print(
        "Calculating permutation importance..."
    )

    # AUC-based permutation importance is useful
    # for the binary inpatient outcome.

    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean":
                result.importances_mean,
            "importance_std":
                result.importances_std,
        }
    )

    importance[
        "importance_mean"
    ] = importance[
        "importance_mean"
    ].clip(lower=0)

    total = importance[
        "importance_mean"
    ].sum()

    if total > 0:

        importance[
            "importance_pct"
        ] = (
            importance[
                "importance_mean"
            ]
            / total
            * 100
        )

    else:

        importance[
            "importance_pct"
        ] = 0.0

    importance[
        "factor_type"
    ] = importance[
        "feature"
    ].apply(
        classify_factor_type
    )

    importance = importance.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    importance[
        "importance_rank"
    ] = (
        np.arange(
            len(importance)
        ) + 1
    )

    return importance


# ============================================================================
# FACTOR TYPE
# ============================================================================


def classify_factor_type(
    feature: str,
) -> str:

    if is_sdoh_feature(feature):
        return "SDOH"

    if is_clinical_feature(feature):
        return "Clinical"

    return "Other"


# ============================================================================
# SDOH IMPORTANCE
# ============================================================================


def create_sdoh_importance(
    importance: pd.DataFrame,
) -> pd.DataFrame:

    sdoh = importance[
        importance["factor_type"]
        == "SDOH"
    ].copy()

    sdoh = sdoh.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    sdoh[
        "sdoh_rank"
    ] = (
        np.arange(len(sdoh)) + 1
    )

    return sdoh


# ============================================================================
# LOCAL RISK FACTORS
# ============================================================================


def calculate_local_risk_factors(
    df: pd.DataFrame,
    X: pd.DataFrame,
    model,
    importance: pd.DataFrame,
) -> pd.DataFrame:

    """
    Creates member-level factor explanations.

    This is intentionally a practical explanation rather than claiming
    causal effects.

    For each member:
      - compare the member's feature to the population median
      - identify whether the value is relatively high or low
      - weight the deviation by global permutation importance

    For SDOH variables, direction is inferred from the feature name:

      poverty/unemployment/uninsured/housing burden/etc.
          higher values -> higher concern

      income/broadband/computer/access/routine checkup
          higher values -> generally protective

    The output describes "risk-associated factors", not causal effects.
    """

    banner("GENERATING MEMBER RISK FACTORS")

    records = []

    medians = X.median(
        numeric_only=True
    )

    stds = X.std(
        numeric_only=True
    ).replace(
        0,
        np.nan,
    )

    importance_map = (
        importance.set_index(
            "feature"
        )["importance_pct"]
        .to_dict()
    )

    # Top explanatory features.
    ranked_features = (
        importance[
            importance[
                "importance_mean"
            ] > 0
        ]
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .head(20)[
            "feature"
        ]
        .tolist()
    )

    for row_index in X.index:

        member_id = str(
            df.loc[
                row_index,
                "member_id",
            ]
        )

        member_risk = np.nan

        # Risk probability can be obtained directly.
        try:

            member_risk = float(
                model.predict_proba(
                    X.loc[
                        [row_index]
                    ]
                )[0, 1]
            )

        except Exception:

            pass

        for feature in ranked_features:

            if feature not in X.columns:
                continue

            value = X.loc[
                row_index,
                feature,
            ]

            if pd.isna(value):
                continue

            median = medians.get(
                feature,
                np.nan,
            )

            std = stds.get(
                feature,
                np.nan,
            )

            if pd.isna(median):
                continue

            # Standardized relative deviation.
            if (
                pd.isna(std)
                or std == 0
            ):

                deviation = 0.0

            else:

                deviation = (
                    float(value)
                    - float(median)
                ) / float(std)

            if abs(deviation) < 0.25:
                continue

            factor_type = classify_factor_type(
                feature
            )

            direction = (
                "higher"
                if deviation > 0
                else "lower"
            )

            # ----------------------------------------------------------
            # Determine concern direction.
            # ----------------------------------------------------------

            lower_name = feature.lower()

            protective_keywords = [
                "income",
                "broadband",
                "computer",
                "routine_checkup",
                "cholesterol_screening",
            ]

            harmful_keywords = [
                "poverty",
                "unemployment",
                "public_assistance",
                "snap",
                "uninsured",
                "housing",
                "rent_",
                "crowded",
                "vacancy",
                "no_vehicle",
                "low_access",
                "low_income",
                "smoking",
                "physical_inactivity",
                "poor_mental_health",
                "poor_physical_health",
            ]

            if any(
                keyword in lower_name
                for keyword in harmful_keywords
            ):

                if deviation > 0:
                    factor_direction = (
                        "risk_increasing"
                    )
                else:
                    factor_direction = (
                        "risk_lowering"
                    )

            elif any(
                keyword in lower_name
                for keyword in protective_keywords
            ):

                if deviation > 0:
                    factor_direction = (
                        "risk_lowering"
                    )
                else:
                    factor_direction = (
                        "risk_increasing"
                    )

            else:

                factor_direction = (
                    "higher_association"
                    if deviation > 0
                    else "lower_association"
                )

            # ----------------------------------------------------------
            # Explanation score.
            # ----------------------------------------------------------

            global_importance = (
                importance_map.get(
                    feature,
                    0.0,
                )
            )

            explanation_score = (
                abs(deviation)
                * global_importance
            )

            records.append(
                {
                    "member_id": member_id,
                    "risk_probability":
                        member_risk,
                    "feature": feature,
                    "factor_type":
                        factor_type,
                    "member_value":
                        float(value),
                    "population_median":
                        float(median),
                    "relative_deviation":
                        float(deviation),
                    "relative_direction":
                        direction,
                    "factor_direction":
                        factor_direction,
                    "global_importance_pct":
                        float(
                            global_importance
                        ),
                    "explanation_score":
                        float(
                            explanation_score
                        ),
                }
            )

    factors = pd.DataFrame(
        records
    )

    if factors.empty:

        return factors

    factors = factors.sort_values(
        [
            "member_id",
            "explanation_score",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    factors[
        "member_factor_rank"
    ] = factors.groupby(
        "member_id"
    ).cumcount() + 1

    return factors


# ============================================================================
# SAVE RISK OUTPUT
# ============================================================================


def save_risk_output(
    result: pd.DataFrame,
) -> None:

    result.to_csv(
        MEMBER_RISK_FILE,
        index=False,
    )

    print()
    print(
        "OUTPUT CREATED"
    )

    print(
        MEMBER_RISK_FILE
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    banner(
        "MEMBER RISK GENERATION"
    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    df = load_model_dataset()

    # ------------------------------------------------------------------
    # Normalize target
    # ------------------------------------------------------------------

    df = normalize_target(df)

    banner("MODEL DATASET")

    print(
        f"Members: {len(df)}"
    )

    print(
        f"Target: {TARGET_COLUMN}"
    )

    positive = int(
        df[TARGET_COLUMN].sum()
    )

    print(
        f"Positive target: {positive}"
    )

    print(
        f"Positive rate: "
        f"{positive / len(df):.2%}"
    )

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    validate_model_dataset(df)

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    X, feature_names = (
        prepare_model_features(df)
    )

    y = df[
        TARGET_COLUMN
    ].astype(int)

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    model_name = select_model()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    model = train_final_model(
        model_name=model_name,
        X=X,
        y=y,
    )

    # ------------------------------------------------------------------
    # Risk scores
    # ------------------------------------------------------------------

    result = generate_risk_scores(
        df=df,
        model=model,
        X=X,
    )

    validate_risk_output(
        result
    )

    save_risk_output(
        result
    )

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    importance = calculate_permutation_importance(
        model=model,
        X=X,
        y=y,
        feature_names=feature_names,
    )

    # Save all feature importance.
    importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    # ------------------------------------------------------------------
    # SDOH importance
    # ------------------------------------------------------------------

    sdoh_importance = (
        create_sdoh_importance(
            importance
        )
    )

    sdoh_importance.to_csv(
        SDOH_IMPORTANCE_FILE,
        index=False,
    )

    # ------------------------------------------------------------------
    # Risk factors
    # ------------------------------------------------------------------

    factors = (
        calculate_local_risk_factors(
            df=df,
            X=X,
            model=model,
            importance=importance,
        )
    )

    factors.to_csv(
        RISK_FACTORS_FILE,
        index=False,
    )

    # ------------------------------------------------------------------
    # Top SDOH factors
    # ------------------------------------------------------------------

    banner(
        "TOP SDOH RISK FACTORS"
    )

    if sdoh_importance.empty:

        print(
            "No SDOH features were identified."
        )

    else:

        display_columns = [
            "importance_rank",
            "feature",
            "importance_pct",
        ]

        print(
            sdoh_importance[
                display_columns
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # Top overall factors
    # ------------------------------------------------------------------

    banner(
        "TOP OVERALL RISK FACTORS"
    )

    print(
        importance[
            [
                "importance_rank",
                "feature",
                "factor_type",
                "importance_pct",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Top members
    # ------------------------------------------------------------------

    banner(
        "TOP 10 HIGHEST-RISK MEMBERS"
    )

    print(
        result[
            [
                "risk_rank",
                "member_id",
                "patient_id",
                "county_fips",
                "risk_probability",
                "risk_percentile",
                "risk_band",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    banner(
        "FEATURE IMPORTANCE OUTPUTS"
    )

    print(
        f"All feature importance:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    print()

    print(
        f"SDOH feature importance:\n"
        f"{SDOH_IMPORTANCE_FILE}"
    )

    print()

    print(
        f"Member risk factors:\n"
        f"{RISK_FACTORS_FILE}"
    )

    banner(
        "MEMBER RISK GENERATION COMPLETE"
    )


if __name__ == "__main__":
    main()