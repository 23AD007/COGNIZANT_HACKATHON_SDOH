from pathlib import Path
import pandas as pd
import numpy as np


RAW_DIR = Path("data/raw/synthea")
OUT_DIR = Path("data/processed")

PATIENTS = RAW_DIR / "patients.csv"
ENCOUNTERS = RAW_DIR / "encounters.csv"
CONDITIONS = RAW_DIR / "conditions.csv"
PROCEDURES = RAW_DIR / "procedures.csv"

OUTPUT = OUT_DIR / "member_outcome_labels.csv"
REPORT = OUT_DIR / "clinical_outcome_label_report.csv"


def load_data():
    patients = pd.read_csv(PATIENTS)
    encounters = pd.read_csv(ENCOUNTERS)
    conditions = pd.read_csv(CONDITIONS)
    procedures = pd.read_csv(PROCEDURES)

    patients["member_id"] = patients["Id"].astype(str)

    encounters["member_id"] = encounters["PATIENT"].astype(str)
    conditions["member_id"] = conditions["PATIENT"].astype(str)
    procedures["member_id"] = procedures["PATIENT"].astype(str)

    return patients, encounters, conditions, procedures


def validate_members(patients, encounters, conditions, procedures):
    patient_ids = set(patients["member_id"])

    for name, df in {
        "encounters": encounters,
        "conditions": conditions,
        "procedures": procedures,
    }.items():

        unknown = set(df["member_id"]) - patient_ids

        if unknown:
            raise ValueError(
                f"{name} contains {len(unknown)} member IDs "
                "that do not exist in patients.csv."
            )

    if patients["member_id"].duplicated().any():
        raise ValueError("patients.csv contains duplicate member IDs.")


def build_encounter_features(patients, encounters):
    result = patients[["member_id"]].copy()

    encounter_counts = (
        encounters
        .groupby("member_id")
        .size()
        .rename("encounter_count")
    )

    result = result.merge(
        encounter_counts,
        on="member_id",
        how="left"
    )

    result["encounter_count"] = result["encounter_count"].fillna(0)

    # Emergency / inpatient use
    emergency_inpatient = (
        encounters.assign(
            emergency_or_inpatient=
            encounters["ENCOUNTERCLASS"]
            .astype(str)
            .str.lower()
            .isin(["emergency", "inpatient"])
        )
        .groupby("member_id")["emergency_or_inpatient"]
        .any()
        .rename("emergency_or_inpatient_use")
    )

    result = result.merge(
        emergency_inpatient,
        on="member_id",
        how="left"
    )

    result["emergency_or_inpatient_use"] = (
        result["emergency_or_inpatient_use"]
        .fillna(False)
        .astype(int)
    )

    return result


def build_condition_features(patients, conditions):
    result = patients[["member_id"]].copy()

    condition_counts = (
        conditions
        .groupby("member_id")
        .size()
        .rename("condition_occurrence_count")
    )

    result = result.merge(
        condition_counts,
        on="member_id",
        how="left"
    )

    result["condition_occurrence_count"] = (
        result["condition_occurrence_count"]
        .fillna(0)
    )

    # Distinct condition codes
    distinct_conditions = (
        conditions
        .dropna(subset=["CODE"])
        .assign(CODE=lambda x: x["CODE"].astype(str))
        .groupby("member_id")["CODE"]
        .nunique()
        .rename("chronic_condition_burden")
    )

    result = result.merge(
        distinct_conditions,
        on="member_id",
        how="left"
    )

    result["chronic_condition_burden"] = (
        result["chronic_condition_burden"]
        .fillna(0)
    )

    return result


def build_procedure_features(patients, procedures):
    result = patients[["member_id"]].copy()

    procedure_counts = (
        procedures
        .groupby("member_id")
        .size()
        .rename("procedure_occurrence_count")
    )

    result = result.merge(
        procedure_counts,
        on="member_id",
        how="left"
    )

    result["procedure_occurrence_count"] = (
        result["procedure_occurrence_count"]
        .fillna(0)
    )

    distinct_procedures = (
        procedures
        .dropna(subset=["CODE"])
        .assign(CODE=lambda x: x["CODE"].astype(str))
        .groupby("member_id")["CODE"]
        .nunique()
        .rename("procedure_burden")
    )

    result = result.merge(
        distinct_procedures,
        on="member_id",
        how="left"
    )

    result["procedure_burden"] = (
        result["procedure_burden"]
        .fillna(0)
    )

    return result


def create_labels(features):
    result = features.copy()

    # Primary target:
    # High utilization = top 25% of observed encounter counts.
    threshold = result["encounter_count"].quantile(0.75)

    result["high_utilization"] = (
        result["encounter_count"] >= threshold
    ).astype(int)

    # Secondary binary clinical utilization target
    result["emergency_or_inpatient_use"] = (
        result["emergency_or_inpatient_use"]
        .astype(int)
    )

    # Composite complexity target.
    #
    # This is intentionally transparent rather than pretending to
    # represent a validated clinical risk score.
    result["complex_care_need"] = (
        (
            result["encounter_count"] >=
            result["encounter_count"].quantile(0.75)
        )
        &
        (
            result["chronic_condition_burden"] >=
            result["chronic_condition_burden"].median()
        )
    ).astype(int)

    return result


def validate_labels(labels, patients):
    if len(labels) != len(patients):
        raise ValueError(
            "Output does not contain exactly one row per patient."
        )

    if labels["member_id"].duplicated().any():
        raise ValueError("Duplicate member_id in outcome labels.")

    if set(labels["member_id"]) != set(patients["member_id"]):
        raise ValueError(
            "Outcome labels do not cover exactly the patients in patients.csv."
        )

    binary_columns = [
        "high_utilization",
        "emergency_or_inpatient_use",
        "complex_care_need",
    ]

    for col in binary_columns:
        values = set(labels[col].dropna().unique())

        if not values.issubset({0, 1}):
            raise ValueError(
                f"{col} contains values other than 0/1: {values}"
            )


def write_report(labels):
    report_rows = []

    outcome_columns = [
        "high_utilization",
        "emergency_or_inpatient_use",
        "chronic_condition_burden",
        "complex_care_need",
        "procedure_burden",
    ]

    for col in outcome_columns:
        series = labels[col]

        report_rows.append({
            "feature": col,
            "rows": len(series),
            "missing": int(series.isna().sum()),
            "unique_values": int(series.nunique()),
            "mean": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
        })

    report = pd.DataFrame(report_rows)
    report.to_csv(REPORT, index=False)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    patients, encounters, conditions, procedures = load_data()

    print("=" * 70)
    print("CLINICAL OUTCOME LABEL CONSTRUCTION")
    print("=" * 70)

    print(f"Patients:    {len(patients)}")
    print(f"Encounters:  {len(encounters)}")
    print(f"Conditions:   {len(conditions)}")
    print(f"Procedures:  {len(procedures)}")

    validate_members(
        patients,
        encounters,
        conditions,
        procedures,
    )

    encounter_features = build_encounter_features(
        patients,
        encounters,
    )

    condition_features = build_condition_features(
        patients,
        conditions,
    )

    procedure_features = build_procedure_features(
        patients,
        procedures,
    )

    features = (
        encounter_features
        .merge(condition_features, on="member_id", how="left")
        .merge(procedure_features, on="member_id", how="left")
    )

    labels = create_labels(features)

    validate_labels(labels, patients)

    # Keep the target table deliberately clean.
    labels = labels[
        [
            "member_id",
            "encounter_count",
            "emergency_or_inpatient_use",
            "condition_occurrence_count",
            "chronic_condition_burden",
            "procedure_occurrence_count",
            "procedure_burden",
            "high_utilization",
            "complex_care_need",
        ]
    ]

    labels.to_csv(OUTPUT, index=False)
    write_report(labels)

    print("\nTARGET DISTRIBUTIONS")
    print("-" * 70)

    for col in [
        "high_utilization",
        "emergency_or_inpatient_use",
        "complex_care_need",
    ]:
        counts = labels[col].value_counts().sort_index()

        print(f"\n{col}")
        print(counts.to_string())

    print("\nCONTINUOUS OUTCOMES")
    print("-" * 70)

    for col in [
        "encounter_count",
        "chronic_condition_burden",
        "procedure_burden",
    ]:
        print(
            f"{col}: "
            f"min={labels[col].min():.0f}, "
            f"median={labels[col].median():.1f}, "
            f"max={labels[col].max():.0f}"
        )

    print("\nOUTPUT")
    print(OUTPUT)

    print("\nREPORT")
    print(REPORT)

    print("\nClinical outcome labels created successfully.")


if __name__ == "__main__":
    main()