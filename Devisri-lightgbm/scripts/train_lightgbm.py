import os
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

# ==================================================
# 1. Folders
# ==================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==================================================
# 2. Load datasets
# ==================================================

features = pd.read_csv(
    "data/processed/member_sdoh_model_features.csv"
)

labels = pd.read_csv(
    "data/processed/member_clinical_target.csv"
)

# Clean column names
features.columns = (
    features.columns
    .str.replace("*", "", regex=False)
    .str.strip()
)

# ==================================================
# 3. Merge
# ==================================================

df = features.merge(
    labels[[
        "patient_id",
        "target_inpatient_any"
    ]],
    left_on="member_id",
    right_on="patient_id",
    how="inner"
)

df = df.drop(columns=["patient_id"])

print("Final dataset shape:", df.shape)

# ==================================================
# 4. Features / Target
# ==================================================

X = df.drop(columns=[
    "member_id",
    "county_fips",
    "target_inpatient_any"
])

y = df["target_inpatient_any"]

print("\nTarget distribution:")
print(y.value_counts())

# ==================================================
# 5. Train/Test split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining:", X_train.shape)
print("Testing :", X_test.shape)

# ==================================================
# 6. Candidate models
# ==================================================

models = {

    "LightGBM_Conservative": lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=10,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    ),

    "LightGBM_Balanced": lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=10,
        max_depth=4,
        min_child_samples=8,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    ),

    "LightGBM_Regularized": lgb.LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.02,
        num_leaves=5,
        max_depth=3,
        min_child_samples=12,
        reg_alpha=0.5,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    )
}

# ==================================================
# 7. Cross Validation
# ==================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision"
}

results = []

print("\n========== MODEL COMPARISON ==========")

for name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    result = {
        "model": name,
        "ROC_AUC": scores["test_roc_auc"].mean(),
        "PR_AUC": scores["test_average_precision"].mean(),
        "F1": scores["test_f1"].mean(),
        "Recall": scores["test_recall"].mean(),
        "Precision": scores["test_precision"].mean()
    }

    results.append(result)

    print("\n", name)
    print("ROC-AUC :", result["ROC_AUC"])
    print("PR-AUC  :", result["PR_AUC"])
    print("F1      :", result["F1"])
    print("Recall  :", result["Recall"])
    print("Precision:", result["Precision"])

# ==================================================
# 8. Save comparison
# ==================================================

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="ROC_AUC",
    ascending=False
)

comparison.to_csv(
    "outputs/lightgbm_model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

# ==================================================
# 9. Select best model
# ==================================================

best_name = comparison.iloc[0]["model"]

best_model = models[best_name]

print("\nBEST MODEL:")
print(best_name)

# ==================================================
# 10. Train best model on training data
# ==================================================

best_model.fit(
    X_train,
    y_train
)

# ==================================================
# 11. Test prediction
# ==================================================

y_prob = best_model.predict_proba(
    X_test
)[:, 1]

# Default threshold
y_pred = (y_prob >= 0.50).astype(int)

# ==================================================
# 12. Test metrics
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("\n========== BEST MODEL TEST PERFORMANCE ==========")

print("Model     :", best_name)
print("Accuracy  :", accuracy)
print("Precision :", precision)
print("Recall    :", recall)
print("F1 Score  :", f1)
print("ROC-AUC   :", roc_auc)
print("PR-AUC    :", pr_auc)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ==================================================
# 13. Save final metrics
# ==================================================

metrics = pd.DataFrame([{
    "model": best_name,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc
}])

metrics.to_csv(
    "outputs/lightgbm_metrics.csv",
    index=False
)

# ==================================================
# 14. Feature importance
# ==================================================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": best_model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    "outputs/lightgbm_feature_importance.csv",
    index=False
)

print("\n========== TOP FEATURES ==========")
print(
    importance.head(20).to_string(index=False)
)

# ==================================================
# 15. Save predictions
# ==================================================

predictions = df.loc[
    X_test.index,
    [
        "member_id",
        "county_fips"
    ]
].copy()

predictions["actual_target"] = y_test
predictions["predicted_target"] = y_pred
predictions["risk_probability"] = y_prob

predictions["risk_level"] = pd.cut(
    y_prob,
    bins=[-0.01, 0.40, 0.70, 1.0],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH"
    ]
)

predictions.to_csv(
    "outputs/lightgbm_predictions.csv",
    index=False
)

# ==================================================
# 16. Save model
# ==================================================

best_model.booster_.save_model(
    "models/lightgbm_model.txt"
)

# ==================================================
# 17. Final output
# ==================================================

print("\n======================================")
print("LIGHTGBM TRAINING COMPLETE")
print("======================================")

print("\nBest model:")
print(best_name)

print("\nSaved files:")
print("models/lightgbm_model.txt")
print("outputs/lightgbm_model_comparison.csv")
print("outputs/lightgbm_metrics.csv")
print("outputs/lightgbm_predictions.csv")
print("outputs/lightgbm_feature_importance.csv")