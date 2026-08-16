import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score


# ============================================================
# 1. DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

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


# ============================================================
# 3. MERGE FEATURES + CLINICAL TARGET
# ============================================================

df = features.merge(
    labels[
        [
            "patient_id",
            "target_inpatient_any"
        ]
    ],
    left_on="member_id",
    right_on="patient_id",
    how="inner"
)

df = df.drop(
    columns=["patient_id"]
)

print(
    "Final dataset shape:",
    df.shape
)


# ============================================================
# 4. NUMERIC CONVERSION
# ============================================================

for col in df.columns:

    if col != "member_id":

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# 5. HELPER FUNCTION
# ============================================================

def get_feature(name):

    if name in df.columns:

        values = df[name].copy()

        median_value = values.median()

        if pd.isna(median_value):

            median_value = 0

        return values.fillna(
            median_value
        )

    return pd.Series(
        0.0,
        index=df.index
    )


# ============================================================
# 6. SDOH VULNERABILITY COMPONENTS
# ============================================================

components = []


# ------------------------------------------------------------
# Economic vulnerability
# ------------------------------------------------------------

components.append(
    get_feature("poverty_pct")
)

components.append(
    get_feature("unemployment_pct")
)

components.append(
    get_feature("public_assistance_pct")
)


# ------------------------------------------------------------
# Income vulnerability
# ------------------------------------------------------------

income = get_feature(
    "median_household_income"
)

income_rank = income.rank(
    pct=True
)

income_vulnerability = (
    1 - income_rank
) * 100

components.append(
    income_vulnerability
)


# ------------------------------------------------------------
# Healthcare access
# ------------------------------------------------------------

components.append(
    get_feature("uninsured_pct")
)

components.append(
    get_feature("places_uninsured_pct")
)


# ------------------------------------------------------------
# Transportation vulnerability
# ------------------------------------------------------------

components.append(
    get_feature("housing_no_vehicle_pct")
)

components.append(
    get_feature("mean_commute_minutes")
)


# ------------------------------------------------------------
# Digital access
# ------------------------------------------------------------

broadband = get_feature(
    "digital_with_broadband_pct"
)

digital_vulnerability = (
    100 - broadband
)

components.append(
    digital_vulnerability
)


# ------------------------------------------------------------
# Housing vulnerability
# ------------------------------------------------------------

components.append(
    get_feature(
        "housing_crowded_1_01_to_1_50_pct"
    )
)

components.append(
    get_feature(
        "housing_crowded_1_51_plus_pct"
    )
)

components.append(
    get_feature(
        "housing_rent_35_plus_pct"
    )
)

components.append(
    get_feature(
        "housing_renter_pct"
    )
)


# ------------------------------------------------------------
# Food / transportation access
# ------------------------------------------------------------

components.append(
    get_feature(
        "driving_low_access_population_beyond_1mi_10mi_count_sum"
    )
)

components.append(
    get_feature(
        "driving_no_vehicle_households_beyond_1mi_count_sum"
    )
)

components.append(
    get_feature(
        "driving_snap_households_beyond_1mi_count_sum"
    )
)


# ------------------------------------------------------------
# Health burden
# ------------------------------------------------------------

health_features = [

    "places_asthma_pct",

    "places_copd_pct",

    "places_diabetes_pct",

    "places_heart_disease_pct",

    "places_obesity_pct",

    "places_physical_inactivity_pct",

    "places_poor_mental_health_pct",

    "places_poor_physical_health_pct",

    "places_smoking_pct",

    "places_stroke_pct"
]


for col in health_features:

    components.append(
        get_feature(col)
    )


# ============================================================
# 7. NORMALIZE COMPONENTS
# ============================================================

normalized_components = []


for component in components:

    minimum = component.min()

    maximum = component.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or maximum == minimum
    ):

        normalized = pd.Series(
            0.0,
            index=df.index
        )

    else:

        normalized = (
            (component - minimum)
            /
            (maximum - minimum)
        )

    normalized_components.append(
        normalized
    )


# ============================================================
# 8. SDOH VULNERABILITY SCORE
# ============================================================

sdoh_score = (

    pd.concat(
        normalized_components,
        axis=1
    )

    .mean(axis=1)

    * 100
)


df[
    "sdoh_vulnerability_score"
] = sdoh_score


# ============================================================
# 9. CLINICAL SCORE
# ============================================================

clinical_score = (

    df[
        "target_inpatient_any"
    ]

    .fillna(0)

    .astype(int)
)


# ============================================================
# 10. GRADED RELEVANCE
# ============================================================
#
# SDOH vulnerability:
#
# 0 - 25   = 0
# 25 - 50  = 1
# 50 - 75  = 2
# 75 - 100 = 3
#
# Clinical inpatient outcome adds +1.
#
# Maximum relevance = 3.
#
# 0 = LOW
# 1 = MEDIUM
# 2 = HIGH
# 3 = CRITICAL
#
# ============================================================

sdoh_level = pd.cut(

    df[
        "sdoh_vulnerability_score"
    ],

    bins=[
        -np.inf,
        25,
        50,
        75,
        np.inf
    ],

    labels=[
        0,
        1,
        2,
        3
    ],

    include_lowest=True
).astype(int)


df["relevance"] = (

    sdoh_level

    +

    clinical_score
)


df["relevance"] = (

    df["relevance"]

    .clip(
        0,
        3
    )

    .astype(int)
)


# ============================================================
# 11. RELEVANCE DISTRIBUTION
# ============================================================

print(
    "\n========== RELEVANCE DISTRIBUTION =========="
)

print(
    df[
        "relevance"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 12. FEATURES
# ============================================================

X = df.drop(

    columns=[

        "member_id",

        "county_fips",

        "target_inpatient_any",

        "relevance",

        "sdoh_vulnerability_score"

    ]
)


y = df[
    "relevance"
]


# ============================================================
# 13. HANDLE MISSING / INFINITE VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


X = X.fillna(
    X.median(
        numeric_only=True
    )
)


X = X.fillna(0)


# ============================================================
# 14. COUNTY GROUPS
# ============================================================

county_groups = (

    df[
        "county_fips"
    ]

    .fillna("UNKNOWN")

    .astype(str)
)


# ============================================================
# 15. TRAIN / TEST SPLIT
# ============================================================
#
# IMPORTANT:
# relevance = 3 has only one member.
# Therefore stratified splitting causes an error.
#
# We use a fixed random split instead.
#
# ============================================================

train_idx, test_idx = train_test_split(

    np.arange(
        len(df)
    ),

    test_size=0.20,

    random_state=42,

    shuffle=True
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


groups_train_raw = county_groups.iloc[
    train_idx
]

groups_test_raw = county_groups.iloc[
    test_idx
]


print(
    "\nTraining:",
    X_train.shape
)

print(
    "Testing :",
    X_test.shape
)


print(
    "\nTraining relevance distribution:"
)

print(
    y_train
    .value_counts()
    .sort_index()
)


print(
    "\nTesting relevance distribution:"
)

print(
    y_test
    .value_counts()
    .sort_index()
)


# ============================================================
# 16. CREATE GROUP SIZES
# ============================================================

def create_group_sizes(series):

    return (

        series

        .value_counts(
            sort=False
        )

        .values

        .tolist()

    )


group_train = create_group_sizes(
    groups_train_raw
)


group_test = create_group_sizes(
    groups_test_raw
)


print(
    "\nTraining groups:",
    len(group_train)
)

print(
    "Testing groups :",
    len(group_test)
)


# ============================================================
# 17. LAMBDAMART MODEL
# ============================================================

model = lgb.LGBMRanker(

    objective="lambdarank",

    metric="ndcg",

    label_gain=[
        0,
        1,
        3,
        7
    ],

    n_estimators=300,

    learning_rate=0.03,

    num_leaves=7,

    max_depth=3,

    min_child_samples=5,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.5,

    reg_lambda=1.0,

    random_state=42,

    verbosity=-1
)

# ============================================================
# 18. TRAIN FINAL LAMBDAMART
# ============================================================

print(
    "\nTraining FINAL graded-relevance LambdaMART..."
)

model = lgb.LGBMRanker(

    objective="lambdarank",

    metric="ndcg",

    label_gain=[
        0,
        1,
        3,
        7
    ],

    n_estimators=300,

    learning_rate=0.03,

    num_leaves=7,

    max_depth=3,

    min_child_samples=5,

    reg_alpha=0.5,

    reg_lambda=1.0,

    random_state=42,

    verbosity=-1
)


# ------------------------------------------------------------
# IMPORTANT
# LightGBM expects evaluation data as a list of datasets.
# For LambdaMART, eval_group must correspond to eval_set.
# ------------------------------------------------------------

model.fit(

    X_train,

    y_train,

    group=group_train,

    eval_set=[
        (X_test, y_test)
    ],

    eval_group=[
        group_test
    ],

    eval_at=[
        1,
        3,
        5,
        10
    ],

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=30,
            verbose=False
        )
    ]
)


# ============================================================
# 19. PREDICT RANKING SCORES
# ============================================================

ranking_scores = model.predict(
    X_test
)


# ============================================================
# 20. CREATE RANKING OUTPUT
# ============================================================

ranking_output = df.iloc[
    test_idx
][

    [

        "member_id",

        "county_fips",

        "target_inpatient_any",

        "sdoh_vulnerability_score",

        "relevance"

    ]

].copy()


ranking_output[
    "ranking_score"
] = ranking_scores


# Highest score = highest priority

ranking_output = (

    ranking_output

    .sort_values(

        "ranking_score",

        ascending=False

    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# 21. PRIORITY RANK
# ============================================================

ranking_output[
    "priority_rank"
] = (

    np.arange(
        len(ranking_output)
    )

    + 1

)


# ============================================================
# 22. PRIORITY LEVEL
# ============================================================

def get_priority(relevance):

    if relevance == 3:

        return "CRITICAL"

    elif relevance == 2:

        return "HIGH"

    elif relevance == 1:

        return "MEDIUM"

    else:

        return "LOW"


ranking_output[
    "priority_level"
] = (

    ranking_output[
        "relevance"
    ]

    .apply(
        get_priority
    )

)


# ============================================================
# 23. NDCG CALCULATION
# ============================================================

ndcg_results = []


for county, group in (

    ranking_output

    .groupby(

        "county_fips",

        dropna=False

    )

):

    if len(group) < 2:

        continue


    true_relevance = np.asarray(

        [

            group[
                "relevance"
            ].values

        ]

    )


    predicted_scores = np.asarray(

        [

            group[
                "ranking_score"
            ].values

        ]

    )


    for k in [
        1,
        3,
        5,
        10
    ]:

        if k <= len(group):

            score = ndcg_score(

                true_relevance,

                predicted_scores,

                k=k

            )


            ndcg_results.append({

                "county_fips":
                    county,

                "k":
                    k,

                "ndcg":
                    score

            })


ndcg_df = pd.DataFrame(
    ndcg_results
)


# ============================================================
# 24. OVERALL NDCG
# ============================================================

overall_ndcg = {}


for k in [
    1,
    3,
    5,
    10
]:

    if len(ndcg_df) == 0:

        overall_ndcg[
            f"NDCG@{k}"
        ] = np.nan

        continue


    valid = ndcg_df[
        ndcg_df[
            "k"
        ] == k
    ]


    if len(valid) > 0:

        overall_ndcg[
            f"NDCG@{k}"
        ] = (

            valid[
                "ndcg"
            ]

            .mean()

        )

    else:

        overall_ndcg[
            f"NDCG@{k}"
        ] = np.nan


# ============================================================
# 25. TOP PRIORITY MEMBERS
# ============================================================

print(
    "\n========== TOP PRIORITY MEMBERS =========="
)


print(

    ranking_output

    .head(20)

    .to_string(
        index=False
    )

)


# ============================================================
# 26. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        X.columns,

    "importance":
        model.feature_importances_

})


importance = (

    importance

    .sort_values(

        "importance",

        ascending=False

    )

)


print(
    "\n========== TOP FEATURES =========="
)


print(

    importance

    .head(20)

    .to_string(
        index=False
    )

)


# ============================================================
# 27. SAVE RANKINGS
# ============================================================

ranking_output.to_csv(

    "outputs/lambdamart_rankings.csv",

    index=False

)


# ============================================================
# 28. SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(

    "outputs/lambdamart_feature_importance.csv",

    index=False

)


# ============================================================
# 29. SAVE NDCG METRICS
# ============================================================

metrics = pd.DataFrame([

    {

        "metric": key,

        "value": value

    }

    for key, value

    in overall_ndcg.items()

])


metrics.to_csv(

    "outputs/lambdamart_metrics.csv",

    index=False

)


# ============================================================
# 30. SAVE RELEVANCE DISTRIBUTION
# ============================================================

relevance_distribution = (

    df[
        "relevance"
    ]

    .value_counts()

    .sort_index()

    .reset_index()

)


relevance_distribution.columns = [

    "relevance",

    "member_count"

]


relevance_distribution.to_csv(

    "outputs/lambdamart_relevance_distribution.csv",

    index=False

)


# ============================================================
# 31. SAVE MEMBER SDOH PRIORITY SCORES
# ============================================================

sdoh_priority = df[

    [

        "member_id",

        "county_fips",

        "sdoh_vulnerability_score",

        "relevance",

        "target_inpatient_any"

    ]

].copy()


sdoh_priority.to_csv(

    "outputs/member_sdoh_priority_scores.csv",

    index=False

)


# ============================================================
# 32. SAVE MODEL
# ============================================================

model.booster_.save_model(

    "models/lambdamart_model.txt"

)


# ============================================================
# 33. PRINT NDCG
# ============================================================

print(
    "\n========== LAMBDAMART NDCG =========="
)


for key, value in overall_ndcg.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# 34. FINAL STATUS
# ============================================================

print(
    "\n=============================================="
)

print(
    "FINAL IMPROVED LAMBDAMART TRAINING COMPLETE"
)

print(
    "=============================================="
)


print(
    "\nSaved files:"
)

print(
    "models/lambdamart_model.txt"
)

print(
    "outputs/lambdamart_rankings.csv"
)

print(
    "outputs/lambdamart_feature_importance.csv"
)

print(
    "outputs/lambdamart_metrics.csv"
)

print(
    "outputs/lambdamart_relevance_distribution.csv"
)

print(
    "outputs/member_sdoh_priority_scores.csv"
)