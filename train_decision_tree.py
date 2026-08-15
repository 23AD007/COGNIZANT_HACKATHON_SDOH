import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

features = pd.read_csv(
    "data/processed/member_sdoh_model_features.csv"
)

target = pd.read_csv(
    "data/processed/member_clinical_target.csv"
)


# ============================================================
# 2. MERGE
# ============================================================

data = features.merge(
    target[["patient_id", "target_inpatient_any"]],
    left_on="member_id",
    right_on="patient_id",
    how="inner"
)

print("Merged data:", data.shape)


# ============================================================
# 3. REMOVE IDENTIFIERS
# ============================================================

data = data.drop(
    columns=["member_id", "patient_id", "county_fips"],
    errors="ignore"
)


# ============================================================
# 4. X AND Y
# ============================================================

X = data.drop(columns=["target_inpatient_any"])
y = data["target_inpatient_any"]

X = X.select_dtypes(include=np.number)


# ============================================================
# 5. IMPUTE MISSING VALUES
# ============================================================

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 7. TRY MULTIPLE DECISION TREES
# ============================================================

models = {

    "Tree_1": DecisionTreeClassifier(
        max_depth=2,
        min_samples_leaf=4,
        class_weight=None,
        random_state=42
    ),

    "Tree_2": DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=3,
        class_weight=None,
        random_state=42
    ),

    "Tree_3": DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=3,
        class_weight=None,
        random_state=42
    ),

    "Tree_4": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=2,
        class_weight=None,
        random_state=42
    ),

    "Tree_5_balanced": DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    ),

    "Tree_6_balanced": DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )
}


# ============================================================
# 8. EVALUATE
# ============================================================

results = []

print("\n==========================================")
print("DECISION TREE COMPARISON")
print("==========================================")

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")


# ============================================================
# 9. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n==========================================")
print("ALL RESULTS")
print("==========================================")

print(
    results_df.to_string(index=False)
)


# ============================================================
# 10. SELECT BEST MODEL BY F1
# ============================================================

best_index = results_df["F1"].idxmax()

best_name = results_df.loc[
    best_index,
    "Model"
]

best_model = models[best_name]

print("\n==========================================")
print("BEST MODEL")
print("==========================================")

print("Selected:", best_name)


# ============================================================
# 11. FINAL PREDICTION
# ============================================================

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

y_probability = best_model.predict_proba(X_test)[:, 1]


print("\n==========================================")
print("FINAL CONFUSION MATRIX")
print("==========================================")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": best_model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\n==========================================")
print("TOP FEATURES")
print("==========================================")

print(
    importance.head(15).to_string(index=False)
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "data/processed/decision_tree_model_comparison.csv",
    index=False
)

importance.to_csv(
    "data/processed/decision_tree_feature_importance.csv",
    index=False
)

print("\nResults saved successfully.")