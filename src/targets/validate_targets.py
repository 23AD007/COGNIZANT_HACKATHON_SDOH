# src/targets/validate_target.py

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "candidate_target_definitions.csv"
OUTPUT = ROOT / "data" / "processed"

OUTPUT.mkdir(parents=True, exist_ok=True)


TARGETS = [
    "target_emergency_any",
    "target_inpatient_any",
    "target_acute_any",
    "target_acute_2plus",
    "target_top25_utilization",
]


def main():

    df = pd.read_csv(INPUT)

    required = [
        "patient_id",
        "encounter_count",
        "emergency_count",
        "inpatient_count",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    missing_targets = [
        c for c in TARGETS
        if c not in df.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Missing target columns: {missing_targets}"
        )

    print("=" * 70)
    print("TARGET VALIDATION")
    print("=" * 70)

    print(f"Members: {len(df)}")

    # ---------------------------------------------------------
    # 1. Class balance
    # ---------------------------------------------------------

    rows = []

    for target in TARGETS:

        positives = int(df[target].sum())
        negatives = len(df) - positives
        positive_pct = positives / len(df) * 100

        rows.append({
            "target": target,
            "positive_members": positives,
            "negative_members": negatives,
            "positive_pct": round(
                positive_pct, 2
            ),
        })

    balance = pd.DataFrame(rows)

    print("\nCLASS BALANCE")
    print(balance.to_string(index=False))

    # ---------------------------------------------------------
    # 2. Target overlap
    # ---------------------------------------------------------

    overlap = pd.DataFrame(
        index=TARGETS,
        columns=TARGETS,
    )

    for a in TARGETS:
        for b in TARGETS:
            overlap.loc[a, b] = int(
                (
                    (df[a] == 1)
                    & (df[b] == 1)
                ).sum()
            )

    overlap = overlap.astype(int)

    print("\nTARGET OVERLAP")
    print(overlap)

    # ---------------------------------------------------------
    # 3. Acute-care distribution
    # ---------------------------------------------------------

    df["acute_count"] = (
        df["emergency_count"]
        + df["inpatient_count"]
    )

    print("\nACUTE-CARE DISTRIBUTION")

    print(
        df["acute_count"]
        .describe()
        .to_string()
    )

    # ---------------------------------------------------------
    # 4. Evaluate target suitability
    # ---------------------------------------------------------

    recommendations = []

    for _, row in balance.iterrows():

        pct = row["positive_pct"]

        if 20 <= pct <= 40:
            assessment = "GOOD_CLASS_BALANCE"

        elif pct < 20:
            assessment = "TOO_RARE"

        elif pct > 60:
            assessment = "TOO_COMMON"

        else:
            assessment = "ACCEPTABLE"

        recommendations.append({
            "target": row["target"],
            "positive_pct": pct,
            "assessment": assessment,
        })

    recommendations = pd.DataFrame(
        recommendations
    )

    print("\nTARGET ASSESSMENT")
    print(
        recommendations.to_string(index=False)
    )

    # ---------------------------------------------------------
    # 5. Select target
    # ---------------------------------------------------------

    # Prefer clinically meaningful acute utilization.
    # Among acute targets, prefer the more selective threshold.

    acute_2plus = balance[
        balance["target"]
        == "target_acute_2plus"
    ].iloc[0]

    if 20 <= acute_2plus["positive_pct"] <= 40:

        selected_target = "target_acute_2plus"

        reason = (
            "Acute-care utilization requiring at least "
            "two emergency/inpatient encounters provides "
            "a clinically interpretable and selective "
            "high-risk target."
        )

    else:

        inpatient = balance[
            balance["target"]
            == "target_inpatient_any"
        ].iloc[0]

        if 20 <= inpatient["positive_pct"] <= 40:

            selected_target = "target_inpatient_any"

            reason = (
                "Any inpatient utilization provides a "
                "clinically meaningful and sufficiently "
                "selective target."
            )

        else:

            selected_target = "target_top25_utilization"

            reason = (
                "The top-quartile utilization definition "
                "is retained as the fallback benchmark."
            )

    # ---------------------------------------------------------
    # 6. Save selected target
    # ---------------------------------------------------------

    selected = df[
        [
            "patient_id",
            "encounter_count",
            "emergency_count",
            "inpatient_count",
            "urgent_care_count",
            *TARGETS,
        ]
    ].copy()

    selected["selected_target"] = (
        selected[selected_target]
    )

    selected["target_definition"] = (
        selected_target
    )

    selected_path = (
        OUTPUT
        / "member_clinical_target.csv"
    )

    selected.to_csv(
        selected_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 7. Save validation report
    # ---------------------------------------------------------

    report = pd.DataFrame({
        "selected_target": [
            selected_target
        ],
        "reason": [
            reason
        ],
        "members": [
            len(df)
        ],
        "positive_members": [
            int(df[selected_target].sum())
        ],
        "negative_members": [
            int(
                len(df)
                - df[selected_target].sum()
            )
        ],
        "positive_pct": [
            round(
                df[selected_target].mean() * 100,
                2,
            )
        ],
    })

    report_path = (
        OUTPUT
        / "target_validation_report.csv"
    )

    report.to_csv(
        report_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("SELECTED TARGET")
    print("=" * 70)

    print(
        f"Target: {selected_target}"
    )

    print(
        f"Positive members: "
        f"{int(df[selected_target].sum())}"
    )

    print(
        f"Positive percentage: "
        f"{df[selected_target].mean() * 100:.2f}%"
    )

    print(f"\nReason: {reason}")

    print("\nOutput:")
    print(selected_path)
    print(report_path)


if __name__ == "__main__":
    main()