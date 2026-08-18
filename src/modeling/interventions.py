from __future__ import annotations

import pandas as pd

from .modeling_config import (
    MEMBER_RISK_FILE,
    FEATURE_FILE,
    INTERVENTION_FILE,
)


INTERVENTION_RULES = {
    "transportation": [
        "housing_no_vehicle_pct",
        "mean_commute_minutes",
        "households_without_vehicle_count_sum",
        "driving_no_vehicle_households_beyond_1mi_count_sum",
        "driving_low_vehicle_access_tract_count",
    ],

    "healthcare_access": [
        "uninsured_pct",
        "places_uninsured_pct",
        "places_routine_checkup_pct",
        "places_cholesterol_screening_pct",
    ],

    "economic_support": [
        "poverty_pct",
        "unemployment_pct",
        "public_assistance_pct",
        "snap_households_count_sum",
        "median_household_income",
    ],

    "digital_access": [
        "digital_with_computer_pct",
        "digital_with_broadband_pct",
    ],

    "housing_support": [
        "housing_vacancy_pct",
        "housing_renter_pct",
        "housing_crowded_1_01_to_1_50_pct",
        "housing_crowded_1_51_plus_pct",
        "housing_rent_30_to_34_9_pct",
        "housing_rent_35_plus_pct",
    ],
}


def normalize_series(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


def main():

    members = pd.read_csv(
        MEMBER_RISK_FILE
    )

    features = pd.read_csv(
        FEATURE_FILE
    )

    data = members.merge(
        features,
        on=[
            "member_id",
            "county_fips",
        ],
        how="left",
    )

    intervention_scores = {}

    for intervention, columns in (
        INTERVENTION_RULES.items()
    ):

        available = [
            c
            for c in columns
            if c in data.columns
        ]

        if not available:
            continue

        component_scores = []

        for column in available:

            values = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            normalized = normalize_series(
                values.fillna(
                    values.median()
                )
            )

            # Risk-driving factors:
            # higher values generally indicate
            # greater burden for these selected variables.
            component_scores.append(
                normalized
            )

        score = pd.concat(
            component_scores,
            axis=1,
        ).mean(axis=1)

        intervention_scores[
            intervention
        ] = score

    result = data[
        [
            "member_id",
            "county_fips",
            "member_risk_score",
            "risk_level",
            "member_rank",
        ]
    ].copy()

    for intervention, score in (
        intervention_scores.items()
    ):
        result[
            f"{intervention}_score"
        ] = score

    score_columns = [
        c
        for c in result.columns
        if c.endswith("_score")
    ]

    long_rows = []

    for _, row in result.iterrows():

        for column in score_columns:

            intervention = column.replace(
                "_score",
                "",
            )

            long_rows.append(
                {
                    "member_id": row[
                        "member_id"
                    ],
                    "county_fips": row[
                        "county_fips"
                    ],
                    "member_risk_score": row[
                        "member_risk_score"
                    ],
                    "intervention": intervention,
                    "intervention_score": row[
                        column
                    ],
                }
            )

    output = pd.DataFrame(
        long_rows
    )

    output["priority_score"] = (
        0.60
        * output["member_risk_score"]
        +
        0.40
        * output["intervention_score"]
    )

    output = output.sort_values(
        "priority_score",
        ascending=False,
    )

    output["priority_rank"] = range(
        1,
        len(output) + 1,
    )

    output.to_csv(
        INTERVENTION_FILE,
        index=False,
    )

    print("=" * 70)
    print("INTERVENTION PRIORITIZATION")
    print("=" * 70)

    print(
        f"Rows: {len(output)}"
    )

    print(
        f"Output: {INTERVENTION_FILE}"
    )


if __name__ == "__main__":
    main()