from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_regression_dataset.csv"
)

RISK_OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member_risk_scores.csv"
)


def main():

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    ID_COLUMNS = [
        "member_id",
        "county_fips",
    ]

    TARGET_COLUMN = "target_inpatient_any"

    # --------------------------------------------------
    # Separate features and target
    # --------------------------------------------------

    X = df.drop(
        columns=ID_COLUMNS + [TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    # --------------------------------------------------
    # Preprocessing + Logistic Regression
    # --------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # --------------------------------------------------
    # Stratified 5-Fold Cross-Validation
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_probability = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    # --------------------------------------------------
    # Classification threshold
    # --------------------------------------------------

    threshold = 0.5

    cv_prediction = (
        cv_probability >= threshold
    ).astype(int)

    # --------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------

    roc_auc = roc_auc_score(
        y,
        cv_probability,
    )

    pr_auc = average_precision_score(
        y,
        cv_probability,
    )

    precision = precision_score(
        y,
        cv_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y,
        cv_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        cv_prediction,
        zero_division=0,
    )

    cm = confusion_matrix(
        y,
        cv_prediction,
    )

    brier_score = brier_score_loss(
        y,
        cv_probability,
    )

    # --------------------------------------------------
    # Train final model on all observed members
    # --------------------------------------------------

    model.fit(X, y)

    # --------------------------------------------------
    # Create member risk output
    # --------------------------------------------------

    risk_output = df[
        [
            "member_id",
            "county_fips",
            TARGET_COLUMN,
        ]
    ].copy()

    # Use out-of-fold probabilities for observed members
    risk_output["risk_probability"] = cv_probability

    risk_output["predicted_risk"] = (
        cv_prediction
    )

    # --------------------------------------------------
    # Save risk scores
    # --------------------------------------------------

    risk_output.to_csv(
        RISK_OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("=" * 60)
    print("LOGISTIC REGRESSION MODEL")
    print("=" * 60)

    print(f"Members: {len(X)}")
    print(f"Features: {X.shape[1]}")

    print("\nTraining completed successfully.")

    # --------------------------------------------------
    # Cross-validation results
    # --------------------------------------------------

    print("\nStratified 5-Fold Cross-Validation")
    print("-" * 40)

    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"PR-AUC:      {pr_auc:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Brier Score: {brier_score:.4f}")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    print("\nConfusion Matrix")
    print("-" * 40)

    print(cm)

    print("\nMatrix format:")
    print("[[TN, FP]")
    print(" [FN, TP]]")

    # --------------------------------------------------
    # Risk score summary
    # --------------------------------------------------

    print("\nRisk Score Summary")
    print("-" * 40)

    print(
        risk_output["risk_probability"]
        .describe()
    )

    print("\nFirst 10 member risk scores:")

    print(
        risk_output.head(10).to_string(
            index=False
        )
    )

    print("\nRisk score output:")
    print(RISK_OUTPUT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()