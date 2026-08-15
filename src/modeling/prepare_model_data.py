from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member_sdoh_model_features.csv"
)

TARGET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "member_clinical_target.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_regression_dataset.csv"
)


def main():

    # Load feature data
    features = pd.read_csv(FEATURE_FILE)

    # Load target data
    target = pd.read_csv(TARGET_FILE)

    # Rename target ID so both datasets use the same key
    target = target.rename(
        columns={"patient_id": "member_id"}
    )

    # Keep only the required target columns
    target = target[
        ["member_id", "target_inpatient_any"]
    ]

    # Merge features with target
    df = features.merge(
        target,
        on="member_id",
        how="inner",
        validate="one_to_one",
    )

    # Basic validation
    print("=" * 60)
    print("LOGISTIC REGRESSION DATASET")
    print("=" * 60)

    print(f"Feature rows: {len(features)}")
    print(f"Target rows: {len(target)}")
    print(f"Merged rows: {len(df)}")

    print("\nTarget distribution:")
    print(df["target_inpatient_any"].value_counts())

    print("\nMissing target values:")
    print(df["target_inpatient_any"].isna().sum())

    print("\nDataset shape:")
    print(df.shape)

    # Save
    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()