from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "synthea"
OUT = ROOT / "data" / "processed" / "clinical_outcome_audit"

OUT.mkdir(parents=True, exist_ok=True)


CLINICAL_FILES = {
    "patients": "patients.csv",
    "encounters": "encounters.csv",
    "conditions": "conditions.csv",
    "observations": "observations.csv",
    "medications": "medications.csv",
    "procedures": "procedures.csv",
    "allergies": "allergies.csv",
    "immunizations": "immunizations.csv",
    "careplans": "careplans.csv",
}


def find_member_column(df):
    candidates = [
        "Id",
        "ID",
        "id",
        "PATIENT",
        "Patient",
        "patient",
        "patient_id",
        "member_id",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def audit_file(name, filename):
    path = RAW / filename

    if not path.exists():
        return {
            "dataset": name,
            "file": filename,
            "exists": False,
            "rows": 0,
            "columns": 0,
            "member_id_column": None,
            "unique_members": 0,
        }

    df = pd.read_csv(path, low_memory=False)

    member_col = find_member_column(df)

    unique_members = (
        df[member_col].nunique(dropna=True)
        if member_col
        else 0
    )

    result = {
        "dataset": name,
        "file": filename,
        "exists": True,
        "rows": len(df),
        "columns": len(df.columns),
        "member_id_column": member_col,
        "unique_members": unique_members,
    }

    # Clinical date/event columns
    date_candidates = [
        c for c in df.columns
        if any(
            token in c.lower()
            for token in [
                "date",
                "start",
                "stop",
                "time",
            ]
        )
    ]

    result["date_columns"] = "|".join(date_candidates)

    return result


def inspect_encounters():
    path = RAW / "encounters.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path, low_memory=False)

    print("\n" + "=" * 70)
    print("ENCOUNTER DATA")
    print("=" * 70)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumns:")
    print("\n".join(map(str, df.columns)))

    for col in [
        "ENCOUNTERCLASS",
        "DESCRIPTION",
        "CODE",
        "REASONCODE",
        "REASONDESCRIPTION",
    ]:
        if col in df.columns:
            print(f"\n{col} value counts:")
            print(
                df[col]
                .value_counts(dropna=False)
                .head(30)
                .to_string()
            )

    return df


def inspect_conditions():
    path = RAW / "conditions.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path, low_memory=False)

    print("\n" + "=" * 70)
    print("CONDITION DATA")
    print("=" * 70)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    for col in [
        "CODE",
        "DESCRIPTION",
    ]:
        if col in df.columns:
            print(f"\n{col} value counts:")
            print(
                df[col]
                .value_counts(dropna=False)
                .head(50)
                .to_string()
            )

    return df


def inspect_procedures():
    path = RAW / "procedures.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path, low_memory=False)

    print("\n" + "=" * 70)
    print("PROCEDURE DATA")
    print("=" * 70)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    for col in [
        "CODE",
        "DESCRIPTION",
    ]:
        if col in df.columns:
            print(f"\n{col} value counts:")
            print(
                df[col]
                .value_counts(dropna=False)
                .head(50)
                .to_string()
            )

    return df


def build_member_coverage():
    """
    Determine which members have clinical records in each dataset.
    """

    coverage = None

    for name, filename in CLINICAL_FILES.items():

        path = RAW / filename

        if not path.exists():
            continue

        df = pd.read_csv(path, low_memory=False)

        member_col = find_member_column(df)

        if not member_col:
            continue

        ids = (
            df[member_col]
            .dropna()
            .astype(str)
            .drop_duplicates()
        )

        current = pd.DataFrame(
            {
                "member_id": ids,
                name: True,
            }
        )

        if coverage is None:
            coverage = current
        else:
            coverage = coverage.merge(
                current,
                on="member_id",
                how="outer",
            )

    if coverage is None:
        return pd.DataFrame()

    coverage = coverage.fillna(False)

    return coverage


def main():

    print("=" * 70)
    print("SYNTHEA CLINICAL OUTCOME AUDIT")
    print("=" * 70)

    print(f"Raw Synthea directory: {RAW}")

    # ---------------------------------------------------------------
    # 1. Dataset inventory
    # ---------------------------------------------------------------

    inventory = pd.DataFrame(
        [
            audit_file(name, filename)
            for name, filename in CLINICAL_FILES.items()
        ]
    )

    print("\n" + "=" * 70)
    print("CLINICAL DATASET INVENTORY")
    print("=" * 70)

    print(
        inventory[
            [
                "dataset",
                "file",
                "exists",
                "rows",
                "columns",
                "member_id_column",
                "unique_members",
            ]
        ].to_string(index=False)
    )

    inventory.to_csv(
        OUT / "clinical_dataset_inventory.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 2. Member coverage
    # ---------------------------------------------------------------

    coverage = build_member_coverage()

    if not coverage.empty:

        print("\n" + "=" * 70)
        print("MEMBER CLINICAL COVERAGE")
        print("=" * 70)

        print(f"Members represented: {len(coverage)}")

        for col in coverage.columns:

            if col == "member_id":
                continue

            count = coverage[col].sum()

            print(
                f"{col:20s}: "
                f"{count:4d} members"
            )

        coverage.to_csv(
            OUT / "member_clinical_coverage.csv",
            index=False,
        )

    # ---------------------------------------------------------------
    # 3. Encounters
    # ---------------------------------------------------------------

    encounters = inspect_encounters()

    if encounters is not None:

        member_col = find_member_column(encounters)

        if member_col:

            encounter_counts = (
                encounters
                .groupby(member_col)
                .size()
                .reset_index(name="encounter_count")
            )

            encounter_counts.to_csv(
                OUT / "member_encounter_counts.csv",
                index=False,
            )

    # ---------------------------------------------------------------
    # 4. Conditions
    # ---------------------------------------------------------------

    conditions = inspect_conditions()

    if conditions is not None:

        member_col = find_member_column(conditions)

        if member_col:

            condition_counts = (
                conditions
                .groupby(member_col)
                .size()
                .reset_index(name="condition_count")
            )

            condition_counts.to_csv(
                OUT / "member_condition_counts.csv",
                index=False,
            )

    # ---------------------------------------------------------------
    # 5. Procedures
    # ---------------------------------------------------------------

    procedures = inspect_procedures()

    if procedures is not None:

        member_col = find_member_column(procedures)

        if member_col:

            procedure_counts = (
                procedures
                .groupby(member_col)
                .size()
                .reset_index(name="procedure_count")
            )

            procedure_counts.to_csv(
                OUT / "member_procedure_counts.csv",
                index=False,
            )

    # ---------------------------------------------------------------
    # 6. Candidate outcome summary
    # ---------------------------------------------------------------

    candidates = []

    if encounters is not None:

        candidates.extend(
            [
                {
                    "candidate_outcome":
                        "encounter_count",
                    "source":
                        "encounters.csv",
                    "type":
                        "utilization",
                    "independent_of_sdoh":
                        True,
                    "available":
                        True,
                },
                {
                    "candidate_outcome":
                        "encounter_class",
                    "source":
                        "encounters.csv",
                    "type":
                        "encounter_type",
                    "independent_of_sdoh":
                        True,
                    "available":
                        "ENCOUNTERCLASS" in encounters.columns,
                },
            ]
        )

    if conditions is not None:

        candidates.append(
            {
                "candidate_outcome":
                    "condition_occurrence",
                "source":
                    "conditions.csv",
                "type":
                    "clinical_condition",
                "independent_of_sdoh":
                    True,
                "available":
                    True,
            }
        )

    if procedures is not None:

        candidates.append(
            {
                "candidate_outcome":
                    "procedure_occurrence",
                "source":
                    "procedures.csv",
                "type":
                    "clinical_procedure",
                "independent_of_sdoh":
                    True,
                "available":
                    True,
            }
        )

    candidate_df = pd.DataFrame(candidates)

    candidate_df.to_csv(
        OUT / "candidate_outcomes.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CANDIDATE OUTCOMES")
    print("=" * 70)

    if candidate_df.empty:
        print("No candidate clinical outcomes identified.")
    else:
        print(candidate_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(f"\nReports written to:")
    print(OUT)


if __name__ == "__main__":
    main()