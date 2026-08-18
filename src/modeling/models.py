"""
Model definitions for SDOH-driven member risk prediction.

Models:
    1. Logistic Regression
    2. Random Forest
    3. LightGBM

The dataset is currently small (~108 members), so the models
are intentionally regularized/conservative to reduce overfitting.

Target:
    target_inpatient_any

Important:
    - member_id is NOT a model feature
    - county/geographic identifiers are NOT model features
    - preprocessing/imputation should happen inside each CV fold
    - all models expose predict_proba()
"""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier


def create_logistic_regression() -> LogisticRegression:
    """
    Create the Logistic Regression baseline.

    Logistic Regression is retained because:
        - it is highly interpretable
        - it provides a strong linear baseline
        - it is useful for checking whether nonlinear models
          are actually adding predictive value
    """

    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )


def create_random_forest() -> RandomForestClassifier:
    """
    Create the Random Forest model.

    Conservative settings are used because the current
    member-level dataset is small.
    """

    return RandomForestClassifier(
        n_estimators=300,

        # Limit tree complexity
        max_depth=5,

        # Prevent leaves based on very few observations
        min_samples_split=6,
        min_samples_leaf=3,

        # Handle binary class imbalance
        class_weight="balanced",

        # Reproducibility
        random_state=42,

        # Use all available CPU cores
        n_jobs=-1,
    )


def create_lightgbm() -> LGBMClassifier:
    """
    Create the LightGBM model.

    The configuration is deliberately conservative because
    the current dataset contains only around 108 members.

    Previous issue:
        UserWarning:
        X does not have valid feature names,
        but LGBMClassifier was fitted with feature names

    This model itself does not convert DataFrames to NumPy arrays.
    The training code must preserve feature names when calling
    fit() and predict_proba().
    """

    return LGBMClassifier(
        # ------------------------------------------------------
        # Binary classification
        # ------------------------------------------------------
        objective="binary",

        # ------------------------------------------------------
        # Boosting configuration
        # ------------------------------------------------------
        n_estimators=200,
        learning_rate=0.03,

        # ------------------------------------------------------
        # Tree complexity
        # ------------------------------------------------------
        num_leaves=7,
        max_depth=3,

        # Minimum observations in a leaf
        min_child_samples=10,

        # ------------------------------------------------------
        # Sampling
        # ------------------------------------------------------
        subsample=0.8,
        colsample_bytree=0.8,

        # ------------------------------------------------------
        # Regularization
        # ------------------------------------------------------
        reg_alpha=1.0,
        reg_lambda=2.0,

        # ------------------------------------------------------
        # Class imbalance
        # ------------------------------------------------------
        class_weight="balanced",

        # ------------------------------------------------------
        # Reproducibility
        # ------------------------------------------------------
        random_state=42,

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------
        verbosity=-1,
    )


def build_models() -> Dict[str, object]:
    """
    Return all candidate models.

    The training pipeline evaluates all three models using
    the same cross-validation folds.

    Returns
    -------
    dict
        Dictionary containing:
            logistic_regression
            random_forest
            lightgbm
    """

    models = {
        "logistic_regression": create_logistic_regression(),
        "random_forest": create_random_forest(),
        "lightgbm": create_lightgbm(),
    }

    return models


if __name__ == "__main__":

    print("=" * 70)
    print("MODEL CONFIGURATION TEST")
    print("=" * 70)

    models = build_models()

    for name, model in models.items():

        print("\n" + "-" * 70)
        print(name)
        print("-" * 70)

        print(model)