from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "encounter_patient_summary.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = ["patient_id", "encounter_count"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    # Identify encounter-class columns produced by the previous script.
    emergency_col = find_column(
        df,
        [
            "encounter_class_emergency",
        ],
    )

    inpatient_col = find_column(
        df,
        [
            "encounter_class_inpatient",
        ],
    )

    urgent_col = find_column(
        df,
        [
            "encounter_class_urgentcare",
            "encounter_class_urgent_care",
        ],
    )

    for col in [emergency_col, inpatient_col, urgent_col]:
        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if emergency_col is None:
        df["emergency_count"] = 0
    else:
        df["emergency_count"] = df[emergency_col]

    if inpatient_col is None:
        df["inpatient_count"] = 0
    else:
        df["inpatient_count"] = df[inpatient_col]

    if urgent_col is None:
        df["urgent_care_count"] = 0
    else:
        df["urgent_care_count"] = df[urgent_col]

    # ---------------------------------------------------------
    # Candidate target definitions
    # ---------------------------------------------------------

    df["target_emergency_any"] = (
        df["emergency_count"] >= 1
    ).astype(int)

    df["target_inpatient_any"] = (
        df["inpatient_count"] >= 1
    ).astype(int)

    df["target_acute_any"] = (
        (df["emergency_count"] >= 1)
        | (df["inpatient_count"] >= 1)
    ).astype(int)

    df["target_acute_2plus"] = (
        (
            df["emergency_count"]
            + df["inpatient_count"]
        ) >= 2
    ).astype(int)

    # Top 25% total utilization.
    utilization_threshold = df["encounter_count"].quantile(0.75)

    df["target_top25_utilization"] = (
        df["encounter_count"] >= utilization_threshold
    ).astype(int)

    # ---------------------------------------------------------
    # Compare definitions
    # ---------------------------------------------------------

    target_columns = [
        "target_emergency_any",
        "target_inpatient_any",
        "target_acute_any",
        "target_acute_2plus",
        "target_top25_utilization",
    ]

    rows = []

    for target in target_columns:
        positive = int(df[target].sum())
        total = len(df)

        rows.append(
            {
                "target": target,
                "members": total,
                "positive_members": positive,
                "negative_members": total - positive,
                "positive_pct": round(
                    positive / total * 100,
                    2,
                ),
                "minimum_encounters_positive": (
                    df.loc[df[target] == 1, "encounter_count"].min()
                ),
                "median_encounters_positive": (
                    df.loc[df[target] == 1, "encounter_count"].median()
                ),
                "maximum_encounters_positive": (
                    df.loc[df[target] == 1, "encounter_count"].max()
                ),
            }
        )

    comparison = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Additional acute-care statistics
    # ---------------------------------------------------------

    acute_summary = pd.DataFrame(
        {
            "metric": [
                "members",
                "emergency_any",
                "inpatient_any",
                "acute_any",
                "acute_2plus",
                "total_emergency_encounters",
                "total_inpatient_encounters",
            ],
            "value": [
                len(df),
                int(df["target_emergency_any"].sum()),
                int(df["target_inpatient_any"].sum()),
                int(df["target_acute_any"].sum()),
                int(df["target_acute_2plus"].sum()),
                int(df["emergency_count"].sum()),
                int(df["inpatient_count"].sum()),
            ],
        }
    )

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    comparison_path = (
        OUTPUT_DIR / "target_definition_comparison.csv"
    )

    member_targets_path = (
        OUTPUT_DIR / "candidate_target_definitions.csv"
    )

    acute_summary_path = (
        OUTPUT_DIR / "acute_care_target_summary.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    df[
        [
            "patient_id",
            "encounter_count",
            "emergency_count",
            "inpatient_count",
            "urgent_care_count",
            *target_columns,
        ]
    ].to_csv(
        member_targets_path,
        index=False,
    )

    acute_summary.to_csv(
        acute_summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Console report
    # ---------------------------------------------------------

    print("=" * 70)
    print("TARGET DEFINITION COMPARISON")
    print("=" * 70)

    print(f"Members: {len(df)}")
    print(
        f"75th-percentile encounter threshold: "
        f"{utilization_threshold:.2f}"
    )

    print("\nEncounter columns:")
    print(f"  Emergency: {emergency_col}")
    print(f"  Inpatient: {inpatient_col}")
    print(f"  Urgent care: {urgent_col}")

    print("\nTarget comparison:")
    print(
        comparison.to_string(index=False)
    )

    print("\nAcute-care summary:")
    print(
        acute_summary.to_string(index=False)
    )

    print("\nOutputs:")
    print(comparison_path)
    print(member_targets_path)
    print(acute_summary_path)


if __name__ == "__main__":
    main()