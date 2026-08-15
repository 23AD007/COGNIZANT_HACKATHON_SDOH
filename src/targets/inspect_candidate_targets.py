from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SYNTHEA_DIR = ROOT / "data" / "raw" / "synthea"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    path = SYNTHEA_DIR / name

    if not path.exists():
        raise FileNotFoundError(f"Missing Synthea file: {path}")

    return pd.read_csv(path, low_memory=False)


def inspect_encounters():
    df = load_csv("encounters.csv")

    print("\n" + "=" * 70)
    print("ENCOUNTERS")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique patients: {df['PATIENT'].nunique():,}")

    if "ENCOUNTERCLASS" in df.columns:
        print("\nEncounter classes:")
        print(df["ENCOUNTERCLASS"].value_counts(dropna=False).to_string())

    if "DESCRIPTION" in df.columns:
        print("\nTop encounter descriptions:")
        print(df["DESCRIPTION"].value_counts().head(20).to_string())

    if "START" in df.columns:
        dates = pd.to_datetime(df["START"], errors="coerce")
        print("\nEncounter date range:")
        print(f"  Start: {dates.min()}")
        print(f"  End:   {dates.max()}")

    return df


def inspect_conditions():
    df = load_csv("conditions.csv")

    print("\n" + "=" * 70)
    print("CONDITIONS")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique patients: {df['PATIENT'].nunique():,}")

    if "DESCRIPTION" in df.columns:
        print("\nTop conditions:")
        print(df["DESCRIPTION"].value_counts().head(30).to_string())

    return df


def inspect_medications():
    df = load_csv("medications.csv")

    print("\n" + "=" * 70)
    print("MEDICATIONS")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Unique patients: {df['PATIENT'].nunique():,}")

    if "DESCRIPTION" in df.columns:
        print("\nTop medications:")
        print(df["DESCRIPTION"].value_counts().head(20).to_string())

    return df


def inspect_procedures():
    df = load_csv("procedures.csv")

    print("\n" + "=" * 70)
    print("PROCEDURES")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Unique patients: {df['PATIENT'].nunique():,}")

    if "DESCRIPTION" in df.columns:
        print("\nTop procedures:")
        print(df["DESCRIPTION"].value_counts().head(20).to_string())

    return df


def build_patient_coverage(encounters, conditions, medications, procedures):
    patient_ids = set()

    for df in [encounters, conditions, medications, procedures]:
        if "PATIENT" in df.columns:
            patient_ids.update(df["PATIENT"].dropna().astype(str))

    coverage = pd.DataFrame({"patient_id": sorted(patient_ids)})

    coverage["encounter_count"] = coverage["patient_id"].map(
        encounters["PATIENT"].astype(str).value_counts()
    ).fillna(0)

    coverage["condition_count"] = coverage["patient_id"].map(
        conditions["PATIENT"].astype(str).value_counts()
    ).fillna(0)

    coverage["medication_count"] = coverage["patient_id"].map(
        medications["PATIENT"].astype(str).value_counts()
    ).fillna(0)

    coverage["procedure_count"] = coverage["patient_id"].map(
        procedures["PATIENT"].astype(str).value_counts()
    ).fillna(0)

    return coverage


def build_encounter_summary(encounters):
    df = encounters.copy()

    df["patient_id"] = df["PATIENT"].astype(str)

    summary = (
        df.groupby("patient_id")
        .size()
        .reset_index(name="encounter_count")
    )

    if "ENCOUNTERCLASS" in df.columns:
        pivot = pd.crosstab(
            df["patient_id"],
            df["ENCOUNTERCLASS"]
        ).reset_index()

        pivot.columns = [
            "patient_id"
            if col == "patient_id"
            else f"encounter_class_{str(col).lower().replace(' ', '_')}"
            for col in pivot.columns
        ]

        summary = summary.merge(
            pivot,
            on="patient_id",
            how="left"
        )

    return summary


def build_candidate_targets(encounters, conditions):
    enc = encounters.copy()
    cond = conditions.copy()

    enc["patient_id"] = enc["PATIENT"].astype(str)
    cond["patient_id"] = cond["PATIENT"].astype(str)

    targets = (
        enc.groupby("patient_id")
        .size()
        .reset_index(name="encounter_count")
    )

    condition_counts = (
        cond.groupby("patient_id")
        .size()
        .reset_index(name="condition_count")
    )

    targets = targets.merge(
        condition_counts,
        on="patient_id",
        how="outer"
    )

    targets[["encounter_count", "condition_count"]] = (
        targets[["encounter_count", "condition_count"]]
        .fillna(0)
    )

    # Candidate threshold only.
    # DO NOT treat this as the final target yet.
    encounter_threshold = targets["encounter_count"].quantile(0.75)

    targets["candidate_high_utilization"] = (
        targets["encounter_count"] >= encounter_threshold
    ).astype(int)

    return targets, encounter_threshold


def main():
    encounters = inspect_encounters()
    conditions = inspect_conditions()
    medications = inspect_medications()
    procedures = inspect_procedures()

    coverage = build_patient_coverage(
        encounters,
        conditions,
        medications,
        procedures
    )

    encounter_summary = build_encounter_summary(encounters)

    candidate_targets, threshold = build_candidate_targets(
        encounters,
        conditions
    )

    coverage.to_csv(
        OUTPUT_DIR / "clinical_patient_coverage.csv",
        index=False
    )

    encounter_summary.to_csv(
        OUTPUT_DIR / "encounter_patient_summary.csv",
        index=False
    )

    candidate_targets.to_csv(
        OUTPUT_DIR / "candidate_targets_raw.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("CANDIDATE TARGET SUMMARY")
    print("=" * 70)

    print(f"Patients with clinical records: {len(coverage):,}")
    print(
        f"Encounter threshold used for inspection: "
        f"{threshold:.2f}"
    )

    print("\nCandidate high-utilization distribution:")
    print(
        candidate_targets[
            "candidate_high_utilization"
        ].value_counts()
        .sort_index()
        .to_string()
    )

    print("\nOutput files:")
    print(
        OUTPUT_DIR / "clinical_patient_coverage.csv"
    )
    print(
        OUTPUT_DIR / "encounter_patient_summary.csv"
    )
    print(
        OUTPUT_DIR / "candidate_targets_raw.csv"
    )

    print("\nIMPORTANT:")
    print(
        "The high-utilization label is only a candidate inspection label."
    )
    print(
        "It must NOT be used for modelling until its definition is validated."
    )


if __name__ == "__main__":
    main()