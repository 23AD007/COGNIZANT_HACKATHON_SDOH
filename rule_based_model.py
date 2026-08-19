import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 1. FILE PATHS
# ============================================================

SDOH_FILE = "data/processed/member_sdoh_model_features.csv"
TARGET_FILE = "data/processed/member_clinical_target.csv"

OUTPUT_FILE = "data/processed/member_sdoh_risk_scores.csv"
DASHBOARD_FILE = "data/processed/sdoh_rule_based_dashboard.png"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n==============================================")
print("LOADING DATA")
print("==============================================")

sdoh_df = pd.read_csv(SDOH_FILE)
target_df = pd.read_csv(TARGET_FILE)

print("SDOH data shape   :", sdoh_df.shape)
print("Target data shape :", target_df.shape)


# ============================================================
# 3. CHECK TARGET COLUMNS
# ============================================================

target_columns = [
    "target_emergency_any",
    "target_inpatient_any",
    "target_acute_any",
    "target_acute_2plus",
    "target_top25_utilization",
    "selected_target",
    "target_definition"
]

print("\nTarget columns available:")

for col in target_columns:
    if col in target_df.columns:
        print("✓", col)
    else:
        print("✗", col)


# ============================================================
# 4. CHECK SELECTED TARGET
# ============================================================

if "selected_target" not in target_df.columns:
    raise ValueError(
        "selected_target column is missing."
    )

if "target_definition" not in target_df.columns:
    raise ValueError(
        "target_definition column is missing."
    )


print("\n==============================================")
print("SELECTED TARGET")
print("==============================================")


print(
    target_df["target_definition"].value_counts()
)


# ============================================================
# 5. CONVERT patient_id → member_id
# ============================================================

target_df = target_df.rename(
    columns={
        "patient_id": "member_id"
    }
)


# ============================================================
# 6. MERGE SDOH + TARGET
# ============================================================

df = sdoh_df.merge(
    target_df[
        [
            "member_id",
            "selected_target",
            "target_definition"
        ]
    ],
    on="member_id",
    how="inner"
)


print("\n==============================================")
print("MERGED DATA")
print("==============================================")

print("Final dataset shape:", df.shape)


if len(df) == 0:

    raise ValueError(
        "No matching member_id values found between "
        "the SDOH and target files."
    )


# ============================================================
# 7. SDOH FEATURES
# ============================================================

required_features = [

    "poverty_pct",

    "unemployment_pct",

    "public_assistance_pct",

    "uninsured_pct",

    "housing_no_vehicle_pct",

    "housing_renter_pct",

    "housing_crowded_1_01_to_1_50_pct",

    "housing_crowded_1_51_plus_pct",

    "housing_rent_30_to_34_9_pct",

    "housing_rent_35_plus_pct",

    "mean_commute_minutes",

    "households_without_vehicle_count_sum",

    "driving_low_access_population_beyond_1mi_10mi_count_sum",

    "driving_no_vehicle_households_beyond_1mi_count_sum",

    "driving_snap_households_beyond_1mi_count_sum",

    "driving_low_income_low_access_tract_count",

    "driving_low_vehicle_access_tract_count",

    "straight_no_vehicle_households_beyond_1mi_count_sum",

    "straight_snap_households_beyond_1mi_count_sum",

    "straight_low_income_low_access_tract_count",

    "straight_low_vehicle_access_tract_count"

]


# ============================================================
# 8. CHECK FEATURES
# ============================================================

missing_features = [

    col

    for col in required_features

    if col not in df.columns

]


if missing_features:

    print("\nMissing SDOH features:")

    for col in missing_features:

        print(" -", col)

    raise ValueError(
        "Some required SDOH columns are missing."
    )


print("\n✓ All SDOH features are available.")


# ============================================================
# 9. PREPROCESSING
# ============================================================

for col in required_features:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


for col in required_features:

    df[col] = df[col].fillna(
        df[col].median()
    )


# ============================================================
# 10. PREPARE TARGET
# ============================================================

df["selected_target"] = pd.to_numeric(
    df["selected_target"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "selected_target"
    ]
)


df["selected_target"] = (
    df["selected_target"]
    .astype(int)
)


# Make sure target is binary

unique_targets = sorted(
    df["selected_target"].unique()
)

print("\nTarget values:", unique_targets)


if not set(unique_targets).issubset({0, 1}):

    raise ValueError(
        "selected_target must contain only 0 and 1."
    )


# ============================================================
# 11. TARGET DISTRIBUTION
# ============================================================

print("\n==============================================")
print("TARGET DISTRIBUTION")
print("==============================================")


print(
    df["selected_target"].value_counts()
)


print("\nTarget percentage:")

print(
    df["selected_target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# ============================================================
# 12. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(

    df,

    test_size=0.20,

    random_state=42,

    stratify=df["selected_target"]

)


train_df = train_df.copy()
test_df = test_df.copy()


print("\n==============================================")
print("TRAIN / TEST SPLIT")
print("==============================================")

print(
    "Total    :", len(df)
)

print(
    "Training :", len(train_df)
)

print(
    "Testing  :", len(test_df)
)


# ============================================================
# 13. RULE-BASED SDOH SCORE
# ============================================================

def calculate_sdoh_score(row):

    score = 0

    # --------------------------------------------------------
    # Poverty
    # --------------------------------------------------------

    if row["poverty_pct"] >= 30:
        score += 2

    elif row["poverty_pct"] >= 20:
        score += 1


    # --------------------------------------------------------
    # Unemployment
    # --------------------------------------------------------

    if row["unemployment_pct"] >= 10:
        score += 2

    elif row["unemployment_pct"] >= 5:
        score += 1


    # --------------------------------------------------------
    # Public assistance
    # --------------------------------------------------------

    if row["public_assistance_pct"] >= 20:
        score += 2

    elif row["public_assistance_pct"] >= 10:
        score += 1


    # --------------------------------------------------------
    # Uninsured
    # --------------------------------------------------------

    if row["uninsured_pct"] >= 15:
        score += 2

    elif row["uninsured_pct"] >= 8:
        score += 1


    # --------------------------------------------------------
    # No vehicle
    # --------------------------------------------------------

    if row["housing_no_vehicle_pct"] >= 20:
        score += 2

    elif row["housing_no_vehicle_pct"] >= 10:
        score += 1


    # --------------------------------------------------------
    # Renter households
    # --------------------------------------------------------

    if row["housing_renter_pct"] >= 50:
        score += 1


    # --------------------------------------------------------
    # Housing crowding
    # --------------------------------------------------------

    if row[
        "housing_crowded_1_01_to_1_50_pct"
    ] >= 10:
        score += 1


    if row[
        "housing_crowded_1_51_plus_pct"
    ] >= 5:
        score += 2


    # --------------------------------------------------------
    # Housing rent burden
    # --------------------------------------------------------

    if row[
        "housing_rent_30_to_34_9_pct"
    ] >= 15:
        score += 1


    if row[
        "housing_rent_35_plus_pct"
    ] >= 30:
        score += 2

    elif row[
        "housing_rent_35_plus_pct"
    ] >= 20:
        score += 1


    # --------------------------------------------------------
    # Long commute
    # --------------------------------------------------------

    if row["mean_commute_minutes"] >= 40:
        score += 2

    elif row["mean_commute_minutes"] >= 30:
        score += 1


    # --------------------------------------------------------
    # Households without vehicle
    # --------------------------------------------------------

    if row[
        "households_without_vehicle_count_sum"
    ] > 0:
        score += 1


    # --------------------------------------------------------
    # Low-access population
    # --------------------------------------------------------

    if row[
        "driving_low_access_population_beyond_1mi_10mi_count_sum"
    ] > 0:
        score += 1


    # --------------------------------------------------------
    # No vehicle + distance
    # --------------------------------------------------------

    if row[
        "driving_no_vehicle_households_beyond_1mi_count_sum"
    ] > 0:
        score += 2


    # --------------------------------------------------------
    # SNAP + distance
    # --------------------------------------------------------

    if row[
        "driving_snap_households_beyond_1mi_count_sum"
    ] > 0:
        score += 1


    # --------------------------------------------------------
    # Low-income low-access tract
    # --------------------------------------------------------

    if row[
        "driving_low_income_low_access_tract_count"
    ] > 0:
        score += 2


    # --------------------------------------------------------
    # Low vehicle access tract
    # --------------------------------------------------------

    if row[
        "driving_low_vehicle_access_tract_count"
    ] > 0:
        score += 2


    # --------------------------------------------------------
    # Straight-line access indicators
    # --------------------------------------------------------

    if row[
        "straight_no_vehicle_households_beyond_1mi_count_sum"
    ] > 0:
        score += 1


    if row[
        "straight_snap_households_beyond_1mi_count_sum"
    ] > 0:
        score += 1


    if row[
        "straight_low_income_low_access_tract_count"
    ] > 0:
        score += 1


    if row[
        "straight_low_vehicle_access_tract_count"
    ] > 0:
        score += 1


    return score


# ============================================================
# 14. CALCULATE SCORES
# ============================================================

train_df["sdoh_risk_score"] = train_df.apply(
    calculate_sdoh_score,
    axis=1
)

test_df["sdoh_risk_score"] = test_df.apply(
    calculate_sdoh_score,
    axis=1
)


print("\n==============================================")
print("RULE SCORE DISTRIBUTION")
print("==============================================")


print(
    train_df["sdoh_risk_score"]
    .describe()
)


# ============================================================
# 15. FIND THRESHOLD USING TRAINING DATA ONLY
# ============================================================

print("\n==============================================")
print("FINDING BEST RULE THRESHOLD")
print("==============================================")


y_train = train_df[
    "selected_target"
]


train_scores = train_df[
    "sdoh_risk_score"
]


max_score = int(
    df["sdoh_risk_score"].max()
    if "sdoh_risk_score" in df.columns
    else max(
        train_df["sdoh_risk_score"].max(),
        test_df["sdoh_risk_score"].max()
    )
)


best_threshold = None
best_train_f1 = -1


for threshold in range(
    1,
    int(
        train_df["sdoh_risk_score"].max()
    ) + 1
):

    train_pred = (
        train_scores >= threshold
    ).astype(int)


    score = f1_score(
        y_train,
        train_pred,
        zero_division=0
    )


    if score > best_train_f1:

        best_train_f1 = score

        best_threshold = threshold


print(
    "Best threshold:",
    best_threshold
)

print(
    "Training F1:",
    round(best_train_f1, 4)
)


# ============================================================
# 16. TEST PREDICTION
# ============================================================

test_df["predicted_risk"] = (

    test_df["sdoh_risk_score"]

    >= best_threshold

).astype(int)


# ============================================================
# 17. NORMALIZED SCORE
# ============================================================

actual_max_score = max(
    train_df["sdoh_risk_score"].max(),
    test_df["sdoh_risk_score"].max(),
    1
)


test_df["normalized_risk_score"] = (

    test_df["sdoh_risk_score"]

    / actual_max_score

)


# ============================================================
# 18. RISK CATEGORY
# ============================================================

def risk_category(score):

    if score >= best_threshold:

        return "High"

    elif score >= best_threshold / 2:

        return "Medium"

    else:

        return "Low"


test_df["risk_category"] = (

    test_df["sdoh_risk_score"]
    .apply(risk_category)

)


# ============================================================
# 19. METRICS
# ============================================================

y_true = test_df[
    "selected_target"
]

y_pred = test_df[
    "predicted_risk"
]

risk_score = test_df[
    "normalized_risk_score"
]


accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# 20. AUC
# ============================================================

if len(
    np.unique(y_true)
) == 2:

    roc_auc = roc_auc_score(
        y_true,
        risk_score
    )

    pr_auc = average_precision_score(
        y_true,
        risk_score
    )

else:

    roc_auc = np.nan

    pr_auc = np.nan


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


tn, fp, fn, tp = cm.ravel()


# ============================================================
# 22. PRINT RESULTS
# ============================================================

print("\n==============================================")
print("FINAL TEST RESULTS")
print("==============================================")


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)


print("\nAccuracy percentage:")

print(
    f"{accuracy * 100:.2f}%"
)


print("\n==============================================")
print("CONFUSION MATRIX")
print("==============================================")


print(cm)


print("\nTN =", tn)
print("FP =", fp)
print("FN =", fn)
print("TP =", tp)


# ============================================================
# 23. ACTUAL VS PREDICTED DISTRIBUTION
# ============================================================

print("\n==============================================")
print("ACTUAL TARGET DISTRIBUTION")
print("==============================================")


print(
    y_true.value_counts()
)


print("\n==============================================")
print("PREDICTED RISK DISTRIBUTION")
print("==============================================")


print(
    y_pred.value_counts()
)


# ============================================================
# 24. RISK DISTRIBUTION
# ============================================================

low_count = (
    test_df["risk_category"]
    == "Low"
).sum()

medium_count = (
    test_df["risk_category"]
    == "Medium"
).sum()

high_count = (
    test_df["risk_category"]
    == "High"
).sum()


print("\n==============================================")
print("RISK CATEGORIES")
print("==============================================")


print("Low    :", low_count)
print("Medium :", medium_count)
print("High   :", high_count)


# ============================================================
# 25. SAVE TEST RESULTS
# ============================================================

output_columns = [

    "member_id",

    "county_fips",

    "selected_target",

    "target_definition",

    "sdoh_risk_score",

    "normalized_risk_score",

    "risk_category",

    "predicted_risk"

]


test_df[
    output_columns
].to_csv(

    OUTPUT_FILE,

    index=False

)


print("\nResults saved to:")

print(
    OUTPUT_FILE
)


# ============================================================
# 26. COLORFUL DASHBOARD
# ============================================================

BLUE = "#2563EB"
GREEN = "#16A34A"
ORANGE = "#EA580C"
PURPLE = "#9333EA"
RED = "#DC2626"
YELLOW = "#CA8A04"

NAVY = "#123B6D"
DARK = "#1F2937"
GREY = "#6B7280"

LIGHT_BLUE = "#EFF6FF"
LIGHT_GREEN = "#F0FDF4"
LIGHT_ORANGE = "#FFF7ED"
LIGHT_PURPLE = "#FAF5FF"
LIGHT_RED = "#FEF2F2"
LIGHT_YELLOW = "#FEFCE8"

WHITE = "#FFFFFF"


fig = plt.figure(
    figsize=(18, 11),
    facecolor="#F5F7FB"
)


# ============================================================
# TITLE
# ============================================================

fig.text(

    0.5,

    0.96,

    "SDOH RULE-BASED RISK PREDICTION DASHBOARD",

    ha="center",

    fontsize=22,

    fontweight="bold",

    color=WHITE,

    bbox=dict(

        boxstyle="round,pad=0.7",

        facecolor=NAVY,

        edgecolor=NAVY

    )

)


fig.text(

    0.5,

    0.92,

    "80% Training | 20% Testing | Target: selected_target",

    ha="center",

    fontsize=11,

    color=GREY

)


# ============================================================
# METRIC CARD FUNCTION
# ============================================================

def metric_card(
    x,
    title,
    value,
    color,
    background
):

    ax = fig.add_axes(
        [x, 0.76, 0.14, 0.10]
    )

    ax.axis("off")

    box = FancyBboxPatch(

        (0, 0),

        1,

        1,

        boxstyle="round,pad=0.02",

        facecolor=background,

        edgecolor=color,

        linewidth=2

    )

    ax.add_patch(box)

    ax.text(
        0.5,
        0.68,
        title,
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=DARK
    )

    ax.text(
        0.5,
        0.30,
        value,
        ha="center",
        fontsize=19,
        fontweight="bold",
        color=color
    )


metric_card(
    0.03,
    "ACCURACY",
    f"{accuracy * 100:.2f}%",
    BLUE,
    LIGHT_BLUE
)


metric_card(
    0.19,
    "PRECISION",
    f"{precision * 100:.2f}%",
    GREEN,
    LIGHT_GREEN
)


metric_card(
    0.35,
    "RECALL",
    f"{recall * 100:.2f}%",
    ORANGE,
    LIGHT_ORANGE
)


metric_card(
    0.51,
    "F1 SCORE",
    f"{f1:.4f}",
    PURPLE,
    LIGHT_PURPLE
)


metric_card(
    0.67,
    "ROC-AUC",
    f"{roc_auc:.4f}",
    RED,
    LIGHT_RED
)


metric_card(
    0.83,
    "THRESHOLD",
    str(best_threshold),
    YELLOW,
    LIGHT_YELLOW
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

ax_cm = fig.add_axes(
    [0.04, 0.43, 0.27, 0.25]
)


ax_cm.imshow(
    cm,
    cmap="RdYlGn"
)


ax_cm.set_title(
    "CONFUSION MATRIX",
    fontsize=14,
    fontweight="bold",
    color=NAVY
)


ax_cm.set_xlabel(
    "Predicted"
)

ax_cm.set_ylabel(
    "Actual"
)


ax_cm.set_xticks([0, 1])

ax_cm.set_yticks([0, 1])

ax_cm.set_xticklabels(
    ["0", "1"]
)

ax_cm.set_yticklabels(
    ["0", "1"]
)


for i in range(2):

    for j in range(2):

        ax_cm.text(

            j,

            i,

            str(cm[i, j]),

            ha="center",

            va="center",

            fontsize=20,

            fontweight="bold"

        )


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

ax_score = fig.add_axes(
    [0.35, 0.43, 0.30, 0.25]
)


ax_score.hist(

    test_df[
        "sdoh_risk_score"
    ],

    bins=10,

    color=PURPLE,

    edgecolor=WHITE

)


ax_score.axvline(

    best_threshold,

    color=RED,

    linestyle="--",

    linewidth=2,

    label="High Risk Threshold"

)


ax_score.set_title(
    "SDOH RISK SCORE DISTRIBUTION",
    fontsize=14,
    fontweight="bold",
    color=NAVY
)


ax_score.set_xlabel(
    "SDOH Risk Score"
)

ax_score.set_ylabel(
    "Members"
)


ax_score.legend(
    frameon=False
)


# ============================================================
# RISK CATEGORY
# ============================================================

ax_pie = fig.add_axes(
    [0.69, 0.41, 0.26, 0.29]
)


risk_values = [
    low_count,
    medium_count,
    high_count
]


risk_labels = [
    "Low",
    "Medium",
    "High"
]


risk_colors = [
    GREEN,
    YELLOW,
    RED
]


ax_pie.pie(

    risk_values,

    labels=risk_labels,

    colors=risk_colors,

    autopct="%1.1f%%",

    startangle=90,

    wedgeprops=dict(
        width=0.42,
        edgecolor=WHITE
    )

)


ax_pie.set_title(
    "RISK CATEGORY DISTRIBUTION",
    fontsize=14,
    fontweight="bold",
    color=NAVY
)


# ============================================================
# INFORMATION CARDS
# ============================================================

info_ax = fig.add_axes(
    [0.04, 0.08, 0.91, 0.27]
)

info_ax.axis("off")


info_ax.text(

    0.02,

    0.90,

    "MODEL SUMMARY",

    fontsize=14,

    fontweight="bold",

    color=NAVY

)


info_ax.text(

    0.02,

    0.72,

    f"Total members       : {len(df)}",

    fontsize=11,

    color=DARK

)


info_ax.text(

    0.02,

    0.60,

    f"Training members    : {len(train_df)}",

    fontsize=11,

    color=DARK

)


info_ax.text(

    0.02,

    0.48,

    f"Testing members     : {len(test_df)}",

    fontsize=11,

    color=DARK

)


info_ax.text(

    0.02,

    0.36,

    f"Selected target     : selected_target",

    fontsize=11,

    color=DARK

)


info_ax.text(

    0.02,

    0.24,

    f"Target definition   : {df['target_definition'].iloc[0]}",

    fontsize=11,

    color=DARK

)


info_ax.text(

    0.02,

    0.12,

    f"Best rule threshold : {best_threshold}",

    fontsize=11,

    color=DARK

)


# ------------------------------------------------------------
# RISK SUMMARY
# ------------------------------------------------------------

info_ax.text(

    0.38,

    0.90,

    "RISK SUMMARY",

    fontsize=14,

    fontweight="bold",

    color=NAVY

)


info_ax.text(

    0.38,

    0.72,

    f"Low risk members    : {low_count}",

    fontsize=11,

    color=GREEN

)


info_ax.text(

    0.38,

    0.58,

    f"Medium risk members : {medium_count}",

    fontsize=11,

    color=YELLOW

)


info_ax.text(

    0.38,

    0.44,

    f"High risk members   : {high_count}",

    fontsize=11,

    color=RED

)


info_ax.text(

    0.38,

    0.28,

    f"True positives      : {tp}",

    fontsize=11,

    color=DARK

)


info_ax.text(

    0.38,

    0.16,

    f"False negatives     : {fn}",

    fontsize=11,

    color=DARK

)


# ------------------------------------------------------------
# KEY MESSAGE
# ------------------------------------------------------------

info_ax.text(

    0.68,

    0.90,

    "PROJECT INTERPRETATION",

    fontsize=14,

    fontweight="bold",

    color=NAVY

)


info_ax.text(

    0.68,

    0.70,

    "SDOH factors are converted",

    fontsize=11,

    color=DARK

)


info_ax.text(

    0.68,

    0.60,

    "into a rule-based risk score.",

    fontsize=11,

    color=DARK

)


info_ax.text(

    0.68,

    0.46,

    "Higher score → greater",

    fontsize=11,

    color=DARK

)


info_ax.text(

    0.68,

    0.36,

    "social risk and intervention priority.",

    fontsize=11,

    color=DARK

)


info_ax.text(

    0.68,

    0.20,

    "Target = selected clinical outcome",

    fontsize=11,

    fontweight="bold",

    color=PURPLE

)


# ============================================================
# SAVE DASHBOARD
# ============================================================

plt.savefig(

    DASHBOARD_FILE,

    dpi=250,

    bbox_inches="tight"

)


plt.show()


print("\n==============================================")
print("DASHBOARD CREATED")
print("==============================================")


print(
    DASHBOARD_FILE
)


print("\nMODEL COMPLETED SUCCESSFULLY.")

