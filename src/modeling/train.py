from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

FEATURE_FILE = PROCESSED_DIR / "member_model_features.csv"
TARGET_FILE = PROCESSED_DIR / "member_clinical_target.csv"

MODEL_COMPARISON_FILE = (
    PROCESSED_DIR / "model_comparison_cv.csv"
)

FOLD_RESULTS_FILE = (
    PROCESSED_DIR / "model_comparison_fold_results.csv"
)

OOF_FILE = (
    PROCESSED_DIR / "model_oof_predictions.csv"
)


# ============================================================
# CONFIG
# ============================================================

TARGET_COLUMN = "target_inpatient_any"

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# COLUMNS THAT MUST NEVER ENTER THE MODEL
# ============================================================

ID_COLUMNS = {
    "member_id",
    "patient_id",
}

GEOGRAPHY_COLUMNS = {
    "county_fips",
    "zip",
    "fips",
    "lat",
    "lon",
    "state_fips",
    "state_fips_acs",
    "county_geoid_acs",
    "tract_geoid_acs",
    "tract_geoid",
    "tract_geoid_places",
}

IDENTIFIER_COLUMNS = {
    "GEO_ID",
    "NAME",
    "geographic_level",
}

DATE_COLUMNS = {
    "index_date",
}

TARGET_COLUMNS = {
    "target_inpatient_any",
    "target_emergency_any",
    "target_acute_any",
    "target_acute_2plus",
    "target_top25_utilization",
    "selected_target",
    "target_definition",
}

LEAKAGE_COLUMNS = {
    "encounter_count",
    "emergency_count",
    "inpatient_count",
    "urgent_care_count",
    "clinical_inpatient_history_count",
    "clinical_encounter_count",
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_model_dataset():
    print("=" * 70)
    print("MODEL DATASET")
    print("=" * 70)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Member model features not found:\n{FEATURE_FILE}"
        )

    df = pd.read_csv(FEATURE_FILE)

    print(f"Members: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    if "member_id" not in df.columns:
        raise ValueError(
            "member_model_features.csv must contain member_id"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    df["member_id"] = (
        df["member_id"]
        .astype(str)
        .str.strip()
    )

    if df["member_id"].duplicated().any():
        duplicates = df.loc[
            df["member_id"].duplicated(),
            "member_id",
        ].tolist()

        raise ValueError(
            f"Duplicate member_id values found: {duplicates[:10]}"
        )

    return df


# ============================================================
# PREPARE MODEL MATRIX
# ============================================================

def prepare_model_matrix(df):
    """
    Convert the merged member dataset into a clean ML matrix.

    Important:
    - county_fips is NOT a predictor.
    - identifiers are NOT predictors.
    - dates are NOT predictors.
    - target/leakage columns are NOT predictors.
    - only numeric predictors are retained.
    """

    excluded = set()

    excluded.update(ID_COLUMNS)
    excluded.update(GEOGRAPHY_COLUMNS)
    excluded.update(IDENTIFIER_COLUMNS)
    excluded.update(DATE_COLUMNS)
    excluded.update(TARGET_COLUMNS)
    excluded.update(LEAKAGE_COLUMNS)

    # Never allow target column to enter X.
    excluded.add(TARGET_COLUMN)

    candidate_columns = [
        c for c in df.columns
        if c not in excluded
    ]

    # Keep numeric predictors only.
    numeric_columns = []

    for col in candidate_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)

    dropped_non_numeric = [
        c for c in candidate_columns
        if c not in numeric_columns
    ]

    if dropped_non_numeric:
        print()
        print("Dropped non-numeric columns:")
        for col in dropped_non_numeric:
            print(f"  {col}")

    if not numeric_columns:
        raise ValueError(
            "No numeric model features remain after exclusions."
        )

    X = df[numeric_columns].copy()

    y = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    if y.isna().any():
        raise ValueError(
            "Target contains missing/non-numeric values."
        )

    y = y.astype(int)

    # Make infinities missing.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    print()
    print("Excluded columns:")

    for col in sorted(excluded):
        if col in df.columns:
            print(f"  {col}")

    print()
    print("Model features:")

    for col in X.columns:
        print(f"  {col}")

    return X, y, list(X.columns)


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df, X, y):
    print()
    print("=" * 70)
    print("VALIDATING MODEL DATASET")
    print("=" * 70)

    print(f"Members: {len(df)}")
    print(f"Features: {X.shape[1]}")

    positives = int(y.sum())
    negatives = int(len(y) - positives)

    print(f"Positive target: {positives}")
    print(f"Negative target: {negatives}")
    print(
        f"Positive rate: "
        f"{positives / len(y) * 100:.2f}%"
    )

    if len(df) != 108:
        warnings.warn(
            f"Expected 108 members, found {len(df)}."
        )

    if df["member_id"].isna().any():
        raise ValueError(
            "member_id contains missing values."
        )

    if df["member_id"].duplicated().any():
        raise ValueError(
            "member_id contains duplicates."
        )

    if y.nunique() != 2:
        raise ValueError(
            f"Target must contain exactly two classes. "
            f"Found: {sorted(y.unique())}"
        )

    # County can legitimately be missing.
    if "county_fips" in df.columns:
        print(
            f"County missing: "
            f"{df['county_fips'].isna().sum()}"
        )

    print("Leakage check: PASSED")


# ============================================================
# FOLD-SAFE IMPUTATION
# ============================================================

def impute_fold(
    X_train,
    X_valid,
):
    """
    Fit imputation only on X_train.

    Critical fix:
    SimpleImputer can drop columns that are entirely
    missing in a training fold.

    We therefore use keep_empty_features=True so that
    the output always has the same number of columns.
    """

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    X_train_imputed = imputer.fit_transform(X_train)
    X_valid_imputed = imputer.transform(X_valid)

    X_train_imputed = pd.DataFrame(
        X_train_imputed,
        index=X_train.index,
        columns=X_train.columns,
    )

    X_valid_imputed = pd.DataFrame(
        X_valid_imputed,
        index=X_valid.index,
        columns=X_valid.columns,
    )

    return (
        X_train_imputed,
        X_valid_imputed,
        imputer,
    )


# ============================================================
# BUILD MODELS
# ============================================================

def build_models():
    models = {}

    models["logistic_regression"] = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )

    models["random_forest"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbosity=-1,
        )

    return models


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities):
    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
    }


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cross_validation(
    X,
    y,
    models,
):
    print()
    print("=" * 70)
    print("MODEL TRAINING AND CROSS-VALIDATION")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_records = []
    oof_records = []

    for model_name, base_model in models.items():

        print()
        print("=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(X, y),
            start=1,
        ):
            print(
                f"\nProcessing fold "
                f"{fold}/{N_SPLITS}..."
            )

            X_train = X.iloc[train_idx].copy()
            X_valid = X.iloc[valid_idx].copy()

            y_train = y.iloc[train_idx].copy()
            y_valid = y.iloc[valid_idx].copy()

            # ------------------------------------------------
            # FOLD-SAFE IMPUTATION
            # ------------------------------------------------

            (
                X_train_imp,
                X_valid_imp,
                imputer,
            ) = impute_fold(
                X_train,
                X_valid,
            )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            model = clone(base_model)

            model.fit(
                X_train_imp,
                y_train,
            )

            probabilities = model.predict_proba(
                X_valid_imp
            )[:, 1]

            metrics = calculate_metrics(
                y_valid,
                probabilities,
            )

            print(
                f"Fold {fold}: "
                f"ROC-AUC={metrics['roc_auc']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"Brier={metrics['brier']:.4f}"
            )

            fold_records.append(
                {
                    "model": model_name,
                    "fold": fold,
                    **metrics,
                }
            )

            # OOF predictions
            for idx, probability in zip(
                valid_idx,
                probabilities,
            ):
                oof_records.append(
                    {
                        "row_index": int(idx),
                        "member_id": None,
                        "model": model_name,
                        "y_true": int(y.iloc[idx]),
                        "predicted_probability": float(
                            probability
                        ),
                        "fold": fold,
                    }
                )

    fold_df = pd.DataFrame(
        fold_records
    )

    oof_df = pd.DataFrame(
        oof_records
    )

    return fold_df, oof_df


# ============================================================
# MODEL SUMMARY
# ============================================================

def summarize_models(fold_df):
    summary = (
        fold_df
        .groupby("model")
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            brier_mean=("brier", "mean"),
        )
        .reset_index()
    )

    # Primary metric = PR-AUC.
    # Higher is better.
    summary = summary.sort_values(
        "pr_auc_mean",
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        np.arange(1, len(summary) + 1),
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MODEL TRAINING AND CROSS-VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_model_dataset()

    # --------------------------------------------------------
    # PREPARE X / Y
    # --------------------------------------------------------

    X, y, feature_names = prepare_model_matrix(
        df
    )

    print()
    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(f"Members: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(
        f"Positive targets: {int(y.sum())}"
    )
    print(
        f"Negative targets: "
        f"{int(len(y) - y.sum())}"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_dataset(
        df,
        X,
        y,
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    models = build_models()

    print()
    print("Cross-validation folds: 5")
    print()
    print("Models:")

    for name in models:
        print(f"  - {name}")

    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    fold_df, oof_df = run_cross_validation(
        X,
        y,
        models,
    )

    # --------------------------------------------------------
    # ADD MEMBER IDs TO OOF
    # --------------------------------------------------------

    row_to_member = (
        df["member_id"]
        .reset_index(drop=True)
        .to_dict()
    )

    oof_df["member_id"] = (
        oof_df["row_index"]
        .map(row_to_member)
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = summarize_models(
        fold_df
    )

    print()
    print("=" * 70)
    print("FINAL CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Primary selection metric: PR-AUC"
    )

    selected_model = summary.iloc[0]["model"]

    print(
        f"Selected model based on CV PR-AUC: "
        f"{selected_model}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    summary.to_csv(
        MODEL_COMPARISON_FILE,
        index=False,
    )

    fold_df.to_csv(
        FOLD_RESULTS_FILE,
        index=False,
    )

    oof_df.to_csv(
        OOF_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print()
    print("Model comparison:")
    print(MODEL_COMPARISON_FILE)

    print()
    print("Fold results:")
    print(FOLD_RESULTS_FILE)

    print()
    print("OOF predictions:")
    print(OOF_FILE)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()