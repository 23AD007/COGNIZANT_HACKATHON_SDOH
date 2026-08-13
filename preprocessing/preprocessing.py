import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

RAW_ZIP = Path("data/raw/synthea_sample_data_csv_latest.zip")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# COMMON CLEANING FUNCTIONS
# ============================================================

def standardize_columns(df):
    """Convert column names to lowercase snake_case."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def clean_dates(df, columns):
    """Convert date columns to pandas datetime."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def clean_ids(df, columns):
    """Keep identifiers as strings and remove surrounding spaces."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()
    return df


def clean_text(df, columns):
    """Clean text columns."""
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()
    return df


def remove_duplicates(df):
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, removed


# ============================================================
# DATASET-SPECIFIC PREPROCESSING
# ============================================================

def preprocess_allergies(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_dates(df, ["start", "stop"])
    df = clean_ids(df, ["patient", "encounter", "code"])
    df = clean_text(
        df,
        [
            "system", "description", "type", "category",
            "description1", "severity1", "description2", "severity2"
        ],
    )

    for column in ["system", "type", "category", "severity1", "severity2"]:
        if column in df.columns:
            df[column] = df[column].str.lower()

    # STOP is empty in this supplied snapshot.
    df["allergy_status"] = np.where(
        df["stop"].isna(), "active", "inactive"
    )

    return df, duplicates


def preprocess_careplans(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_dates(df, ["start", "stop"])
    df = clean_ids(df, ["id", "patient", "encounter", "code", "reasoncode"])
    df = clean_text(df, ["description", "reasondescription"])

    # No original columns are removed.
    df["careplan_status"] = np.where(
        df["stop"].isna(), "active", "completed"
    )

    return df, duplicates


def preprocess_claims(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    date_columns = [
        "currentillnessdate",
        "servicedate",
        "lastbilleddate1",
        "lastbilleddate2",
        "lastbilleddatep",
    ]
    df = clean_dates(df, date_columns)

    id_columns = [
        "id", "patientid", "providerid",
        "primarypatientinsuranceid", "secondarypatientinsuranceid",
        "departmentid", "patientdepartmentid",
        "referringproviderid", "appointmentid",
        "supervisingproviderid",
        "healthcareclaimtypeid1", "healthcareclaimtypeid2",
    ]
    df = clean_ids(df, id_columns)

    diagnosis_columns = [f"diagnosis{i}" for i in range(1, 9)]
    df = clean_text(df, diagnosis_columns + ["status1", "status2", "statusp"])

    for column in ["outstanding1", "outstanding2", "outstandingp"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Useful claim-level feature.
    df["diagnosis_count"] = df[diagnosis_columns].notna().sum(axis=1)

    return df, duplicates


def preprocess_medications(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_dates(df, ["start", "stop"])
    df = clean_ids(df, ["patient", "payer", "encounter", "code", "reasoncode"])
    df = clean_text(df, ["description", "reasondescription"])

    for column in ["base_cost", "payer_coverage", "dispenses", "totalcost"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["medication_status"] = np.where(
        df["stop"].isna(), "active", "completed"
    )
    df["duration_days"] = (df["stop"] - df["start"]).dt.days

    return df, duplicates


def preprocess_conditions(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_dates(df, ["start", "stop"])
    df = clean_ids(df, ["patient", "encounter", "code"])
    df = clean_text(df, ["system", "description"])

    df["condition_status"] = np.where(
        df["stop"].isna(), "active", "resolved"
    )
    df["duration_days"] = (df["stop"] - df["start"]).dt.days

    return df, duplicates


def preprocess_encounters(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_dates(df, ["start", "stop"])
    df = clean_ids(
        df,
        ["id", "patient", "organization", "provider",
         "payer", "code", "reasoncode"],
    )
    df = clean_text(
        df,
        ["encounterclass", "description", "reasondescription"],
    )

    for column in ["base_encounter_cost", "total_claim_cost", "payer_coverage"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["duration_hours"] = (
        (df["stop"] - df["start"]).dt.total_seconds() / 3600
    )

    return df, duplicates


def preprocess_immunizations(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_dates(df, ["date"])
    df = clean_ids(df, ["patient", "encounter", "code"])
    df = clean_text(df, ["description"])

    if "base_cost" in df.columns:
        df["base_cost"] = pd.to_numeric(df["base_cost"], errors="coerce")

    return df, duplicates


def preprocess_observations(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_dates(df, ["date"])
    df = clean_ids(df, ["patient", "encounter", "code"])
    df = clean_text(df, ["category", "description", "value", "units", "type"])

    # Preserve the original value and create a numeric version where possible.
    df["value_numeric"] = pd.to_numeric(df["value"], errors="coerce")

    return df, duplicates


def preprocess_organizations(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_ids(df, ["id", "zip"])
    df = clean_text(df, ["name", "address", "city", "state", "phone"])

    for column in ["lat", "lon", "revenue", "utilization"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Keep location fields because they are useful for GIS/PostGIS.
    keep = [
        "id", "name", "city", "state", "zip",
        "lat", "lon", "revenue", "utilization"
    ]
    df = df[[c for c in keep if c in df.columns]]

    return df, duplicates


def preprocess_patients(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_dates(df, ["birthdate", "deathdate"])
    df = clean_ids(df, ["id", "zip", "fips"])

    df = clean_text(
        df,
        [
            "prefix", "first", "middle", "last", "suffix", "maiden",
            "marital", "race", "ethnicity", "gender", "birthplace",
            "address", "city", "state", "county", "drivers", "passport"
        ],
    )

    for column in [
        "healthcare_expenses",
        "healthcare_coverage",
        "income",
        "lat",
        "lon",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Analytical dataset: remove direct personal identifiers.
    keep = [
        "id", "birthdate", "deathdate", "marital",
        "race", "ethnicity", "gender", "birthplace",
        "city", "state", "county", "fips", "zip",
        "lat", "lon", "healthcare_expenses",
        "healthcare_coverage", "income"
    ]
    df = df[[c for c in keep if c in df.columns]]

    return df, duplicates


def preprocess_payer_transitions(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_dates(df, ["start_date", "end_date"])
    df = clean_ids(df, ["patient", "memberid", "payer", "secondary_payer"])
    df = clean_text(df, ["plan_ownership", "owner_name"])

    df["transition_duration_days"] = (
        df["end_date"] - df["start_date"]
    ).dt.days

    return df, duplicates


def preprocess_payers(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_ids(df, ["id", "zip"])
    df = clean_text(
        df,
        ["name", "ownership", "address", "city", "state_headquartered", "phone"],
    )

    numeric_columns = [
        "amount_covered", "amount_uncovered", "revenue",
        "covered_encounters", "uncovered_encounters",
        "covered_medications", "uncovered_medications",
        "covered_procedures", "uncovered_procedures",
        "covered_immunizations", "uncovered_immunizations",
        "unique_customers", "qols_avg", "member_months",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df, duplicates


def preprocess_providers(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_ids(df, ["id", "organization", "zip"])
    df = clean_text(
        df,
        ["name", "gender", "speciality", "address", "city", "state"],
    )

    for column in ["lat", "lon", "encounters", "procedures"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Keep useful provider/access fields.
    keep = [
        "id", "organization", "name", "gender", "speciality",
        "city", "state", "zip", "lat", "lon",
        "encounters", "procedures"
    ]
    df = df[[c for c in keep if c in df.columns]]

    return df, duplicates


def preprocess_supplies(df):
    df = standardize_columns(df)
    df, duplicates = remove_duplicates(df)

    df = clean_dates(df, ["date"])
    df = clean_ids(df, ["patient", "encounter", "code"])
    df = clean_text(df, ["description"])

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    return df, duplicates


# ============================================================
# MAIN PIPELINE
# ============================================================

PROCESSORS = {
    "allergies.csv": preprocess_allergies,
    "careplans.csv": preprocess_careplans,
    "claims.csv": preprocess_claims,
    "medications.csv": preprocess_medications,
    "conditions.csv": preprocess_conditions,
    "encounters.csv": preprocess_encounters,
    "immunizations.csv": preprocess_immunizations,
    "observations.csv": preprocess_observations,
    "organizations.csv": preprocess_organizations,
    "patients.csv": preprocess_patients,
    "payer_transitions.csv": preprocess_payer_transitions,
    "payers.csv": preprocess_payers,
    "providers.csv": preprocess_providers,
    "supplies.csv": preprocess_supplies,
}


def main():
    if not RAW_ZIP.exists():
        raise FileNotFoundError(
            f"Raw ZIP not found: {RAW_ZIP.resolve()}\n"
            "Put the Cognizant/Synthea ZIP in the project folder."
        )

    summary = []

    with zipfile.ZipFile(RAW_ZIP, "r") as z:
        available_files = {
            Path(name).name.lower(): name
            for name in z.namelist()
        }

        for filename, processor in PROCESSORS.items():
            if filename not in available_files:
                print(f"[SKIPPED] {filename} not found")
                continue

            with z.open(available_files[filename]) as file:
                raw = pd.read_csv(file)

            processed, duplicates_removed = processor(raw)

            output_file = OUTPUT_DIR / filename.replace(
                ".csv", "_preprocessed.csv"
            )
            processed.to_csv(output_file, index=False)

            patient_column = None
            for candidate in ["patient", "patientid"]:
                if candidate in processed.columns:
                    patient_column = candidate
                    break

            unique_patients = (
                processed[patient_column].nunique()
                if patient_column
                else None
            )

            summary.append({
                "dataset": filename.replace(".csv", ""),
                "original_rows": len(raw),
                "original_columns": len(raw.columns),
                "processed_rows": len(processed),
                "processed_columns": len(processed.columns),
                "duplicates_removed": duplicates_removed,
                "unique_patients": unique_patients,
            })

            print(f"[DONE] {filename} -> {output_file}")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        OUTPUT_DIR / "preprocessing_summary.csv",
        index=False
    )

    print("\n================ PREPROCESSING SUMMARY ================")
    print(summary_df.to_string(index=False))
    print("\nProcessed files are in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()