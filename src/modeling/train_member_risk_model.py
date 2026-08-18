"""
HealthLens Member Risk Model
=============================

Train the member-level clinical + SDOH risk model using the actual
prepared project dataset:

    data/processed/member_model_features.csv

Target:
    target_inpatient_any

Dataset characteristics currently expected:
    108 members
    31 positive
    77 negative

The upstream project pipeline already performs:
    SDOH construction
    clinical feature construction
    target construction
    SDOH + clinical merging
    leakage validation

This module is responsible for:
    1. Loading the prepared model dataset
    2. Building a numeric feature matrix
    3. Removing identifiers / leakage columns
    4. Removing constant predictors
    5. Median imputation
    6. Model comparison
    7. Cross-validation
    8. Final model training
    9. Artifact serialization
   10. Prediction helpers

Public API
----------
load_dataset
load_feature_schema
validate_dataset
build_feature_matrix
train_pipeline
predict_member_risk
save_artifacts
load_artifacts

Run:
    py -3.12 -m src.modeling.train_member_risk_model
"""

from __future__ import annotations

import json
import pickle
import warnings

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore")


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROCESSED_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATASET_FILE = (
    PROCESSED_DIR
    / "member_model_features.csv"
)

DATASET_ALIAS_FILE = (
    PROCESSED_DIR
    / "member_model_dataset.csv"
)

MODEL_FILE = (
    MODEL_DIR
    / "member_risk_model.pkl"
)

SCHEMA_FILE = (
    MODEL_DIR
    / "member_risk_feature_schema.json"
)

METRICS_FILE = (
    MODEL_DIR
    / "member_risk_model_metrics.json"
)

PREDICTIONS_FILE = (
    PROCESSED_DIR
    / "member_risk_model_predictions.csv"
)


# ============================================================================
# MODEL CONSTANTS
# ============================================================================

MODEL_VERSION = "2.0.0"

TARGET_COLUMN = (
    "target_inpatient_any"
)

MEMBER_ID_COLUMNS = {
    "member_id",
    "patient_id",
    "Id",
    "id",
    "NAME",
    "name",
}

NON_PREDICTOR_COLUMNS = {
    # identifiers
    "member_id",
    "patient_id",
    "Id",
    "id",

    # target
    "target_inpatient_any",

    # explicit leakage fields
    "first_inpatient_date",
    "index_date",
    "first_inpatient",
    "inpatient_target",
    "target",

    # graph/geography identifiers
    "GEO_ID",
    "tract_geoid",
    "tract_geoid_acs",
    "tract_geoid_places",
    "county_geoid",
    "county_geoid_acs",
    "county_geoid_places",
    "state_fips",
    "state_fips_acs",
    "state_fips_places",

    # textual identifiers
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
    "geographic_level",
    "places_year",
    "prevalence_type",
    "_merge",
}

# These are explicitly excluded because they represent the target event
# or information derived from the target event.
LEAKAGE_COLUMNS = {
    "target_inpatient_any",
    "first_inpatient_date",
    "index_date",
    "clinical_inpatient_history_count",
}

RANDOM_STATE = 42

CV_FOLDS = 5


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ModelArtifact:
    """
    Serializable trained-model container.
    """

    model_version: str
    model_name: str
    pipeline: Any
    feature_columns: list[str]
    target_column: str
    training_rows: int
    positive_count: int
    negative_count: int
    metrics: dict[str, Any]


# ============================================================================
# DATASET LOADING
# ============================================================================

def _resolve_dataset_path(
    path: str | Path | None = None,
) -> Path:
    """
    Resolve the real prepared member model dataset.

    Priority:
        1. explicitly supplied path
        2. member_model_features.csv
        3. member_model_dataset.csv
    """

    if path is not None:

        candidate = Path(path)

        if candidate.exists():
            return candidate

        raise FileNotFoundError(
            "Requested member model dataset does not exist:\n"
            f"{candidate}"
        )

    if DATASET_FILE.exists():
        return DATASET_FILE

    if DATASET_ALIAS_FILE.exists():
        return DATASET_ALIAS_FILE

    raise FileNotFoundError(
        "Member model dataset not found.\n\n"
        "Expected one of:\n"
        f"  {DATASET_FILE}\n"
        f"  {DATASET_ALIAS_FILE}\n"
    )


def load_dataset(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load the actual prepared member-level model dataset.
    """

    dataset_path = _resolve_dataset_path(
        path
    )

    df = pd.read_csv(
        dataset_path
    )

    if df.empty:
        raise ValueError(
            "Member model dataset is empty:\n"
            f"{dataset_path}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            "Target column missing from member model dataset:\n"
            f"{TARGET_COLUMN}"
        )

    return df


# ============================================================================
# FEATURE SCHEMA
# ============================================================================

def _candidate_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Determine usable predictor columns.

    The prepared dataset contains a mixture of:
        - identifiers
        - textual demographics
        - SDOH predictors
        - clinical predictors
        - target/leakage columns

    Only numeric predictors are retained by the model.
    """

    candidates: list[str] = []

    for column in df.columns:

        if column in NON_PREDICTOR_COLUMNS:
            continue

        if column in LEAKAGE_COLUMNS:
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        # Keep columns with at least one numeric observation.
        if numeric.notna().any():
            candidates.append(
                column
            )

    return candidates


def load_feature_schema(
    path: str | Path | None = None,
) -> list[str]:
    """
    Load the feature schema.

    If an artifact schema exists, use it.

    Otherwise construct the schema from the current dataset.
    """

    schema_path = (
        Path(path)
        if path is not None
        else SCHEMA_FILE
    )

    if schema_path.exists():

        with schema_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            payload = json.load(
                handle
            )

        if isinstance(
            payload,
            dict,
        ):

            features = payload.get(
                "feature_columns"
            )

            if isinstance(
                features,
                list,
            ):

                return [
                    str(x)
                    for x in features
                ]

    df = load_dataset()

    return _candidate_feature_columns(
        df
    )


# ============================================================================
# TARGET VALIDATION
# ============================================================================

def _normalize_target(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Normalize the binary target.
    """

    values = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    if values.isna().any():

        raise ValueError(
            "Target contains missing/non-numeric values."
        )

    unique = set(
        values.astype(int).unique()
    )

    if not unique.issubset(
        {0, 1}
    ):

        raise ValueError(
            "Target must contain only 0/1 values. "
            f"Found: {sorted(unique)}"
        )

    return values.astype(int)


def validate_dataset(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate the prepared model dataset.
    """

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    target = _normalize_target(
        df
    )

    positive = int(
        (target == 1).sum()
    )

    negative = int(
        (target == 0).sum()
    )

    if positive == 0:
        raise ValueError(
            "No positive target examples."
        )

    if negative == 0:
        raise ValueError(
            "No negative target examples."
        )

    features = _candidate_feature_columns(
        df
    )

    if not features:
        raise ValueError(
            "No numeric predictor columns found."
        )

    return {
        "rows": int(len(df)),
        "features_available": int(
            len(features)
        ),
        "positive": positive,
        "negative": negative,
        "positive_rate": (
            positive / len(df)
        ),
    }


# ============================================================================
# FEATURE MATRIX
# ============================================================================

def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build X/y from the prepared dataset.

    Constant predictors are removed.

    Missing numeric values remain NaN here and are handled by the
    model pipeline's SimpleImputer.
    """

    if feature_columns is None:

        feature_columns = (
            _candidate_feature_columns(
                df
            )
        )

    if not feature_columns:
        raise ValueError(
            "No feature columns supplied."
        )

    existing = [
        column
        for column in feature_columns
        if column in df.columns
    ]

    if not existing:
        raise ValueError(
            "None of the requested feature columns exist."
        )

    X = pd.DataFrame(
        index=df.index
    )

    for column in existing:

        X[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove completely missing columns.
    completely_missing = [
        column
        for column in X.columns
        if X[column].notna().sum() == 0
    ]

    if completely_missing:

        X = X.drop(
            columns=completely_missing
        )

    # Remove constant columns.
    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(
            dropna=True
        ) <= 1
    ]

    if constant_columns:

        X = X.drop(
            columns=constant_columns
        )

    if X.shape[1] == 0:
        raise ValueError(
            "No usable numeric predictors remain after validation."
        )

    y = _normalize_target(
        df
    )

    return (
        X,
        y,
        list(X.columns),
    )


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def _build_models() -> dict[str, Any]:
    """
    Build candidate classifiers.

    LightGBM is used when installed.

    Otherwise the pipeline remains functional using sklearn models.
    """

    models: dict[str, Any] = {

        "logistic_regression":
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
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
            ),

        "random_forest":
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
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
            ),
    }

    # Optional LightGBM backend.
    try:

        from lightgbm import LGBMClassifier

        models["lightgbm"] = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
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
                        subsample=0.9,
                        colsample_bytree=0.9,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        verbosity=-1,
                    ),
                ),
            ]
        )

    except ImportError:
        pass

    return models


# ============================================================================
# CROSS VALIDATION
# ============================================================================

def _safe_auc(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:

    try:
        return float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )
    except ValueError:
        return float("nan")


def _cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int = CV_FOLDS,
) -> dict[str, Any]:
    """
    Stratified cross-validation.
    """

    minimum_class = int(
        y.value_counts().min()
    )

    actual_folds = min(
        folds,
        minimum_class,
    )

    if actual_folds < 2:

        raise ValueError(
            "Not enough samples per class for cross-validation."
        )

    splitter = StratifiedKFold(
        n_splits=actual_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    metrics: list[dict[str, float]] = []

    for train_idx, test_idx in splitter.split(
        X,
        y,
    ):

        fold_model = clone(
            model
        )

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        fold_model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            fold_model.predict_proba(
                X_test
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        metrics.append(
            {
                "roc_auc": _safe_auc(
                    y_test,
                    probabilities,
                ),

                "pr_auc":
                    float(
                        average_precision_score(
                            y_test,
                            probabilities,
                        )
                    ),

                "precision":
                    float(
                        precision_score(
                            y_test,
                            predictions,
                            zero_division=0,
                        )
                    ),

                "recall":
                    float(
                        recall_score(
                            y_test,
                            predictions,
                            zero_division=0,
                        )
                    ),

                "f1":
                    float(
                        f1_score(
                            y_test,
                            predictions,
                            zero_division=0,
                        )
                    ),

                "brier":
                    float(
                        brier_score_loss(
                            y_test,
                            probabilities,
                        )
                    ),
            }
        )

    frame = pd.DataFrame(
        metrics
    )

    return {
        "folds": actual_folds,

        "roc_auc_mean":
            float(
                frame["roc_auc"].mean()
            ),

        "roc_auc_std":
            float(
                frame["roc_auc"].std(
                    ddof=0
                )
            ),

        "pr_auc_mean":
            float(
                frame["pr_auc"].mean()
            ),

        "pr_auc_std":
            float(
                frame["pr_auc"].std(
                    ddof=0
                )
            ),

        "precision_mean":
            float(
                frame["precision"].mean()
            ),

        "recall_mean":
            float(
                frame["recall"].mean()
            ),

        "f1_mean":
            float(
                frame["f1"].mean()
            ),

        "brier_mean":
            float(
                frame["brier"].mean()
            ),

        "fold_results":
            metrics,
    }


# ============================================================================
# ARTIFACT SERIALIZATION
# ============================================================================

def save_artifacts(
    artifact: ModelArtifact,
) -> None:
    """
    Save model, schema, and metrics.
    """

    with MODEL_FILE.open(
        "wb"
    ) as handle:

        pickle.dump(
            artifact,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    schema_payload = {
        "model_version":
            artifact.model_version,

        "model_name":
            artifact.model_name,

        "target_column":
            artifact.target_column,

        "feature_columns":
            artifact.feature_columns,

        "feature_count":
            len(artifact.feature_columns),

        "training_rows":
            artifact.training_rows,

        "positive_count":
            artifact.positive_count,

        "negative_count":
            artifact.negative_count,
    }

    with SCHEMA_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            schema_payload,
            handle,
            indent=2,
        )

    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            artifact.metrics,
            handle,
            indent=2,
        )


def load_artifacts(
    path: str | Path | None = None,
) -> ModelArtifact:
    """
    Load trained model artifacts.
    """

    model_path = (
        Path(path)
        if path is not None
        else MODEL_FILE
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "Trained member-risk model not found:\n"
            f"{model_path}"
        )

    with model_path.open(
        "rb"
    ) as handle:

        artifact = pickle.load(
            handle
        )

    if not isinstance(
        artifact,
        ModelArtifact,
    ):

        raise TypeError(
            "Invalid member-risk model artifact."
        )

    return artifact


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_pipeline(
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Complete member-risk training pipeline.
    """

    print("=" * 70)
    print("HEALTHLENS MEMBER RISK MODEL")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    df = load_dataset(
        dataset_path
    )

    validation = validate_dataset(
        df
    )

    print(
        f"Dataset:       {len(df)} members"
    )

    print(
        f"Positive:      {validation['positive']}"
    )

    print(
        f"Negative:      {validation['negative']}"
    )

    print(
        f"Positive rate: "
        f"{validation['positive_rate']:.2%}"
    )

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------

    X, y, feature_columns = (
        build_feature_matrix(
            df
        )
    )

    print(
        f"Model features: {len(feature_columns)}"
    )

    # ------------------------------------------------------------------
    # Candidate models
    # ------------------------------------------------------------------

    models = _build_models()

    comparison: list[dict[str, Any]] = []

    print()
    print("=" * 70)
    print("MODEL TRAINING AND CROSS-VALIDATION")
    print("=" * 70)

    for name, model in models.items():

        print()
        print(
            f"MODEL: {name}"
        )

        metrics = _cross_validate_model(
            model,
            X,
            y,
            CV_FOLDS,
        )

        comparison.append(
            {
                "model": name,
                **{
                    key: value
                    for key, value in metrics.items()
                    if key != "fold_results"
                },
            }
        )

        print(
            f"ROC-AUC: "
            f"{metrics['roc_auc_mean']:.4f}"
        )

        print(
            f"PR-AUC:  "
            f"{metrics['pr_auc_mean']:.4f}"
        )

        print(
            f"F1:      "
            f"{metrics['f1_mean']:.4f}"
        )

        print(
            f"Brier:   "
            f"{metrics['brier_mean']:.4f}"
        )

    # ------------------------------------------------------------------
    # Select model
    #
    # PR-AUC is the primary selection metric because the target is
    # imbalanced (31 positive / 77 negative).
    # ------------------------------------------------------------------

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df = (
        comparison_df
        .sort_values(
            by=[
                "pr_auc_mean",
                "roc_auc_mean",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    selected_name = str(
        comparison_df.iloc[
            0
        ]["model"]
    )

    selected_metrics = next(
        item
        for item in comparison
        if item["model"]
        == selected_name
    )

    print()
    print("=" * 70)
    print("MODEL SELECTION")
    print("=" * 70)

    print(
        f"Selected model: {selected_name}"
    )

    print(
        f"CV PR-AUC: "
        f"{selected_metrics['pr_auc_mean']:.4f}"
    )

    # ------------------------------------------------------------------
    # Train final model
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING FINAL MEMBER-RISK MODEL")
    print("=" * 70)

    final_model = clone(
        models[selected_name]
    )

    final_model.fit(
        X,
        y,
    )

    artifact = ModelArtifact(
        model_version=MODEL_VERSION,
        model_name=selected_name,
        pipeline=final_model,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        training_rows=len(df),
        positive_count=int(
            (y == 1).sum()
        ),
        negative_count=int(
            (y == 0).sum()
        ),
        metrics={
            "selected_model":
                selected_name,

            "comparison":
                comparison,

            "selected_metrics":
                selected_metrics,
        },
    )

    save_artifacts(
        artifact
    )

    # ------------------------------------------------------------------
    # Training predictions
    # ------------------------------------------------------------------

    probabilities = (
        final_model.predict_proba(
            X
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    prediction_df = pd.DataFrame(
        {
            "member_id":
                (
                    df["member_id"]
                    if "member_id"
                    in df.columns
                    else np.arange(
                        len(df)
                    )
                ),

            "target_inpatient_any":
                y.to_numpy(),

            "risk_probability":
                probabilities,

            "risk_prediction":
                predictions,
        }
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print()
    print(
        f"Training rows: {len(df)}"
    )

    print(
        f"Features:      {len(feature_columns)}"
    )

    print(
        f"Model:         {selected_name}"
    )

    print(
        f"Model artifact:\n{MODEL_FILE}"
    )

    print(
        f"Schema:\n{SCHEMA_FILE}"
    )

    print(
        f"Metrics:\n{METRICS_FILE}"
    )

    print(
        f"Predictions:\n{PREDICTIONS_FILE}"
    )

    return {
        "model": artifact,
        "comparison": comparison,
        "selected_model": selected_name,
        "feature_columns": feature_columns,
        "validation": validation,
        "predictions": prediction_df,
    }


# ============================================================================
# MEMBER PREDICTION
# ============================================================================

def predict_member_risk(
    member_row: pd.Series | dict[str, Any],
    artifact: ModelArtifact | None = None,
) -> dict[str, Any]:
    """
    Predict risk for one member.

    The supplied member row must contain the trained feature columns.
    """

    if artifact is None:

        artifact = load_artifacts()

    if isinstance(
        member_row,
        pd.Series,
    ):

        row = member_row.to_dict()

    else:

        row = dict(
            member_row
        )

    X = pd.DataFrame(
        [
            {
                column:
                    pd.to_numeric(
                        row.get(
                            column,
                            np.nan
                        ),
                        errors="coerce",
                    )
                for column
                in artifact.feature_columns
            }
        ]
    )

    probability = float(
        artifact.pipeline
        .predict_proba(X)[0, 1]
    )

    prediction = int(
        probability >= 0.5
    )

    if probability >= 0.80:
        risk_band = "Very High"

    elif probability >= 0.60:
        risk_band = "High"

    elif probability >= 0.40:
        risk_band = "Moderate"

    elif probability >= 0.20:
        risk_band = "Low"

    else:
        risk_band = "Very Low"

    member_id = row.get(
        "member_id"
    )

    if member_id is None:
        member_id = row.get(
            "patient_id"
        )

    return {
        "member_id":
            str(member_id)
            if member_id is not None
            else None,

        "risk_probability":
            probability,

        "risk_prediction":
            prediction,

        "risk_band":
            risk_band,

        "model_version":
            artifact.model_version,

        "model_name":
            artifact.model_name,
    }


# ============================================================================
# SELF TEST
# ============================================================================

def _self_test() -> None:

    print("=" * 70)
    print(
        "HEALTHLENS MEMBER RISK MODEL"
    )
    print("=" * 70)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    df = load_dataset()

    print(
        "Dataset loading:             PASS"
    )

    validation = validate_dataset(
        df
    )

    print(
        "Dataset validation:          PASS"
    )

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    X, y, feature_columns = (
        build_feature_matrix(
            df
        )
    )

    assert len(X) == len(df)

    assert len(y) == len(df)

    assert feature_columns

    print(
        "Feature matrix:              PASS"
    )

    print(
        "Feature schema:              PASS"
    )

    # ------------------------------------------------------------------
    # Public schema API
    # ------------------------------------------------------------------

    schema = load_feature_schema()

    assert schema

    print(
        "load_feature_schema():       PASS"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    result = train_pipeline()

    assert result[
        "selected_model"
    ]

    assert result[
        "feature_columns"
    ]

    print(
        "Training pipeline:           PASS"
    )

    # ------------------------------------------------------------------
    # Artifact
    # ------------------------------------------------------------------

    artifact = load_artifacts()

    assert artifact.pipeline is not None

    assert artifact.feature_columns

    print(
        "Model artifact:              PASS"
    )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    first_row = df.iloc[0]

    prediction = predict_member_risk(
        first_row,
        artifact,
    )

    assert (
        0.0
        <= prediction["risk_probability"]
        <= 1.0
    )

    assert prediction[
        "risk_band"
    ]

    print(
        "Member prediction:           PASS"
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRAINING RESULT"
    )
    print("=" * 70)

    print(
        f"Members:          "
        f"{validation['rows']}"
    )

    print(
        f"Features:         "
        f"{len(feature_columns)}"
    )

    print(
        f"Positive targets: "
        f"{validation['positive']}"
    )

    print(
        f"Negative targets: "
        f"{validation['negative']}"
    )

    print(
        f"Selected model:   "
        f"{result['selected_model']}"
    )

    print(
        f"Test member:      "
        f"{prediction['member_id']}"
    )

    print(
        f"Risk probability: "
        f"{prediction['risk_probability']:.6f}"
    )

    print(
        f"Risk band:        "
        f"{prediction['risk_band']}"
    )

    print()
    print(
        "MODEL SELF-TEST: PASSED"
    )
    print("=" * 70)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "ModelArtifact",
    "MODEL_VERSION",
    "TARGET_COLUMN",
    "DATASET_FILE",
    "MODEL_FILE",
    "SCHEMA_FILE",
    "METRICS_FILE",
    "load_dataset",
    "load_feature_schema",
    "validate_dataset",
    "build_feature_matrix",
    "train_pipeline",
    "predict_member_risk",
    "save_artifacts",
    "load_artifacts",
]


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    _self_test()