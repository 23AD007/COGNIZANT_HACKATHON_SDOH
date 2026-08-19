from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SDOH_FILE = PROCESSED_DIR / "member_sdoh_model_features.csv"
CLINICAL_FILE = PROCESSED_DIR / "member_clinical_features.csv"
TARGET_FILE = PROCESSED_DIR / "member_clinical_target.csv"

OUTPUT_FILE = PROCESSED_DIR / "member_modeling_dataset.csv"


def main():

    print("=" * 70)
    print("BUILDING MEMBER MODELING DATASET")
    print("=" * 70)

    sdoh = pd.read_csv(SDOH_FILE)
    clinical = pd.read_csv(CLINICAL_FILE)
    target = pd.read_csv(TARGET_FILE)

    print(f"SDOH rows:      {len(sdoh)}")
    print(f"Clinical rows:  {len(clinical)}")
    print(f"Target rows:    {len(target)}")

    # ---------------------------------------------------------
    # Validate identifiers
    # ---------------------------------------------------------

    for name, df in [
        ("SDOH", sdoh),
        ("Clinical", clinical),
        ("Target", target),
    ]:

        if "member_id" not in df.columns:
            raise ValueError(
                f"{name} dataset does not contain member_id"
            )

    # ---------------------------------------------------------
    # Remove duplicate member records
    # ---------------------------------------------------------

    sdoh = sdoh.drop_duplicates("member_id")
    clinical = clinical.drop_duplicates("member_id")
    target = target.drop_duplicates("member_id")

    # ---------------------------------------------------------
    # Merge
    # ---------------------------------------------------------

    df = sdoh.merge(
        clinical,
        on="member_id",
        how="left",
        suffixes=("", "_clinical"),
    )

    df = df.merge(
        target,
        on="member_id",
        how="inner",
        suffixes=("", "_target"),
    )

    print(f"\nFinal rows: {len(df)}")

    # ---------------------------------------------------------
    # Check target
    # ---------------------------------------------------------

    if "target_inpatient_any" not in df.columns:
        raise ValueError(
            "target_inpatient_any was not found after merging."
        )

    print("\nTarget distribution:")
    print(df["target_inpatient_any"].value_counts(dropna=False))

    # ---------------------------------------------------------
    # Prevent accidental leakage
    # ---------------------------------------------------------

    leakage_candidates = [
        col
        for col in df.columns
        if any(
            token in col.lower()
            for token in [
                "target",
                "future",
                "outcome",
            ]
        )
        and col != "target_inpatient_any"
    ]

    if leakage_candidates:

        print("\nPotential leakage columns:")
        for col in leakage_candidates:
            print(f"  {col}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCreated:")
    print(OUTPUT_FILE)

    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")


if __name__ == "__main__":
    main()