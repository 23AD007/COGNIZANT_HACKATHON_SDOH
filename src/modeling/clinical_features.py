"""
Build member-level clinical history features from Synthea.

Purpose
-------
Create one clinical feature row per synthetic member so that the
SDOH-only model can later be compared with an SDOH + clinical-history
model.

Important leakage rule
----------------------
The target is:

    target_inpatient_any

Clinical features must not include the inpatient encounter that
defines the positive target.

For positive members:
    clinical history = events BEFORE the first inpatient encounter.

For negative members:
    clinical history = events through the member's last observed
    clinical date.

This produces a member-level historical utilization profile.

Raw clinical sources supported
------------------------------
patients.csv
encounters.csv
conditions.csv
observations.csv
medications.csv
procedures.csv
allergies.csv
immunizations.csv
careplans.csv

Output
------
data/processed/member_clinical_features.csv
data/processed/clinical_feature_catalog.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "synthea_combined"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = (
    PROCESSED_DIR
    / "member_clinical_features.csv"
)

CATALOG_FILE = (
    PROCESSED_DIR
    / "clinical_feature_catalog.csv"
)


# ============================================================
# SOURCE FILES
# ============================================================

PATIENTS_FILE = RAW_DIR / "patients.csv"
ENCOUNTERS_FILE = RAW_DIR / "encounters.csv"
CONDITIONS_FILE = RAW_DIR / "conditions.csv"
OBSERVATIONS_FILE = RAW_DIR / "observations.csv"
MEDICATIONS_FILE = RAW_DIR / "medications.csv"
PROCEDURES_FILE = RAW_DIR / "procedures.csv"
ALLERGIES_FILE = RAW_DIR / "allergies.csv"
IMMUNIZATIONS_FILE = RAW_DIR / "immunizations.csv"
CAREPLANS_FILE = RAW_DIR / "careplans.csv"


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = {
    "patients": [
        "Id",
    ],
    "encounters": [
        "PATIENT",
        "START",
        "ENCOUNTERCLASS",
    ],
    "conditions": [
        "PATIENT",
        "START",
    ],
    "observations": [
        "PATIENT",
        "DATE",
    ],
    "medications": [
        "PATIENT",
        "START",
    ],
    "procedures": [
        "PATIENT",
        "START",
    ],
    "allergies": [
        "PATIENT",
        "START",
    ],
    "immunizations": [
        "PATIENT",
        "DATE",
    ],
    "careplans": [
        "PATIENT",
        "START",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def check_file(path: Path) -> None:
    """
    Check that a required source file exists.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required clinical file not found:\n{path}"
        )


def check_columns(
    df: pd.DataFrame,
    required: list[str],
    dataset_name: str,
) -> None:
    """
    Validate required columns.
    """

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required "
            f"columns: {missing}"
        )


def read_csv(
    path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Read and validate a clinical CSV.
    """

    check_file(path)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    check_columns(
        df,
        REQUIRED_COLUMNS[dataset_name],
        dataset_name,
    )

    return df


def parse_date(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Convert a clinical date column to timezone-naive UTC.

    Synthea files can contain timestamps such as:

        2025-07-22T16:12:20Z

    which pandas may interpret as timezone-aware UTC.

    We normalize everything to UTC and then remove timezone
    information so all clinical date comparisons use the same
    datetime64[ns] representation.
    """

    df = df.copy()

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    return df


def member_count(
    df: pd.DataFrame,
    patient_column: str = "PATIENT",
) -> pd.Series:
    """
    Count rows per member.
    """

    return (
        df.groupby(patient_column)
        .size()
        .rename("count")
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_clinical_data():

    print("=" * 70)
    print("LOADING SYNTHEA CLINICAL DATA")
    print("=" * 70)

    patients = read_csv(
        PATIENTS_FILE,
        "patients",
    )

    encounters = read_csv(
        ENCOUNTERS_FILE,
        "encounters",
    )

    conditions = read_csv(
        CONDITIONS_FILE,
        "conditions",
    )

    observations = read_csv(
        OBSERVATIONS_FILE,
        "observations",
    )

    medications = read_csv(
        MEDICATIONS_FILE,
        "medications",
    )

    procedures = read_csv(
        PROCEDURES_FILE,
        "procedures",
    )

    allergies = read_csv(
        ALLERGIES_FILE,
        "allergies",
    )

    immunizations = read_csv(
        IMMUNIZATIONS_FILE,
        "immunizations",
    )

    careplans = read_csv(
        CAREPLANS_FILE,
        "careplans",
    )

    print(
        f"Patients:       {len(patients):,}"
    )

    print(
        f"Encounters:     {len(encounters):,}"
    )

    print(
        f"Conditions:     {len(conditions):,}"
    )

    print(
        f"Observations:   {len(observations):,}"
    )

    print(
        f"Medications:    {len(medications):,}"
    )

    print(
        f"Procedures:     {len(procedures):,}"
    )

    print(
        f"Allergies:      {len(allergies):,}"
    )

    print(
        f"Immunizations:  {len(immunizations):,}"
    )

    print(
        f"Careplans:      {len(careplans):,}"
    )

    return {
        "patients": patients,
        "encounters": encounters,
        "conditions": conditions,
        "observations": observations,
        "medications": medications,
        "procedures": procedures,
        "allergies": allergies,
        "immunizations": immunizations,
        "careplans": careplans,
    }


# ============================================================
# PREPARE DATES
# ============================================================

def prepare_dates(data):

    data["encounters"] = parse_date(
        data["encounters"],
        "START",
    )

    data["encounters"]["STOP"] = (
        pd.to_datetime(
            data["encounters"]["STOP"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
    )

    data["conditions"] = parse_date(
        data["conditions"],
        "START",
    )

    data["observations"] = parse_date(
        data["observations"],
        "DATE",
    )

    data["medications"] = parse_date(
        data["medications"],
        "START",
    )

    data["procedures"] = parse_date(
        data["procedures"],
        "START",
    )

    data["allergies"] = parse_date(
        data["allergies"],
        "START",
    )

    data["immunizations"] = parse_date(
        data["immunizations"],
        "DATE",
    )

    data["careplans"] = parse_date(
        data["careplans"],
        "START",
    )

    return data


# ============================================================
# DETERMINE MEMBER INDEX DATE
# ============================================================

def build_member_index_dates(
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
):
    """
    Determine the clinical observation cutoff per synthetic member.

    Identifier model
    ----------------
    In the current synthetic population, every Synthea patient
    corresponds to exactly one synthetic member.

        Synthea patients.Id
            -> patient_id
            -> member_id

    We therefore preserve BOTH identifiers rather than renaming
    one identifier downstream.

    Positive member:
        first inpatient encounter date.

    Negative member:
        last observed clinical encounter date.

    The target inpatient encounter itself is excluded from
    predictor history.
    """

    # --------------------------------------------------------
    # Normalize patient identifiers
    # --------------------------------------------------------

    patients = patients.copy()
    encounters = encounters.copy()

    patients["Id"] = (
        patients["Id"]
        .astype("string")
        .str.strip()
    )

    encounters["PATIENT"] = (
        encounters["PATIENT"]
        .astype("string")
        .str.strip()
    )

    # --------------------------------------------------------
    # Validate patient uniqueness
    # --------------------------------------------------------

    if patients["Id"].duplicated().any():

        duplicates = (
            patients.loc[
                patients["Id"].duplicated(
                    keep=False
                ),
                "Id",
            ]
            .dropna()
            .unique()
        )

        raise ValueError(
            "Synthea patients.csv contains duplicate "
            f"patient IDs.\nExamples: {duplicates[:10]}"
        )

    # --------------------------------------------------------
    # Build patient -> member mapping
    #
    # For this synthetic population:
    #
    #     patient_id == member_id
    #
    # but we keep both columns explicitly.
    # --------------------------------------------------------

    patient_member_mapping = pd.DataFrame(
        {
            "patient_id": patients["Id"],
            "member_id": patients["Id"],
        }
    )

    # --------------------------------------------------------
    # Validate mapping
    # --------------------------------------------------------

    if patient_member_mapping[
        "patient_id"
    ].duplicated().any():

        raise ValueError(
            "patient_id is not unique in "
            "patient/member mapping."
        )

    if patient_member_mapping[
        "member_id"
    ].duplicated().any():

        raise ValueError(
            "member_id is not unique in "
            "patient/member mapping."
        )

    # --------------------------------------------------------
    # First inpatient encounter
    # --------------------------------------------------------

    inpatient = encounters[
        encounters["ENCOUNTERCLASS"]
        .astype("string")
        .str.lower()
        .eq("inpatient")
    ].copy()

    first_inpatient = (
        inpatient
        .groupby("PATIENT")["START"]
        .min()
        .rename(
            "first_inpatient_date"
        )
    )

    # --------------------------------------------------------
    # Last observed clinical encounter
    # --------------------------------------------------------

    last_encounter = (
        encounters
        .groupby("PATIENT")["START"]
        .max()
        .rename(
            "last_encounter_date"
        )
    )

    # --------------------------------------------------------
    # Build index table
    # --------------------------------------------------------

    index_df = patient_member_mapping.copy()

    # --------------------------------------------------------
    # Attach first inpatient date
    # --------------------------------------------------------

    index_df = index_df.merge(
        first_inpatient,
        left_on="patient_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Attach last clinical date
    # --------------------------------------------------------

    index_df = index_df.merge(
        last_encounter,
        left_on="patient_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    index_df["target_inpatient_any"] = (
        index_df[
            "first_inpatient_date"
        ]
        .notna()
        .astype(int)
    )

    # --------------------------------------------------------
    # Index date
    #
    # Positive:
    #     first inpatient date
    #
    # Negative:
    #     last observed clinical date
    # --------------------------------------------------------

    index_df["index_date"] = np.where(
        index_df[
            "target_inpatient_any"
        ].eq(1),
        index_df[
            "first_inpatient_date"
        ],
        index_df[
            "last_encounter_date"
        ],
    )

    # --------------------------------------------------------
    # Normalize final datetime type
    # --------------------------------------------------------

    index_df["index_date"] = (
        pd.to_datetime(
            index_df["index_date"],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
    )

    index_df[
        "first_inpatient_date"
    ] = (
        pd.to_datetime(
            index_df[
                "first_inpatient_date"
            ],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
    )

    index_df[
        "last_encounter_date"
    ] = (
        pd.to_datetime(
            index_df[
                "last_encounter_date"
            ],
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(index_df) != len(patients):

        raise ValueError(
            "Patient/member index size changed unexpectedly.\n"
            f"Patients: {len(patients)}\n"
            f"Index rows: {len(index_df)}"
        )

    if index_df[
        "patient_id"
    ].nunique() != len(index_df):

        raise ValueError(
            "patient_id is not unique in index table."
        )

    if index_df[
        "member_id"
    ].nunique() != len(index_df):

        raise ValueError(
            "member_id is not unique in index table."
        )

    print("\n" + "=" * 70)
    print("PATIENT → MEMBER MAPPING")
    print("=" * 70)

    print(
        f"Patients: {len(index_df)}"
    )

    print(
        f"Unique patient IDs: "
        f"{index_df['patient_id'].nunique()}"
    )

    print(
        f"Unique member IDs: "
        f"{index_df['member_id'].nunique()}"
    )

    print(
        "Mapping type: "
        "1 patient → 1 synthetic member"
    )

    print(
        "Patient/member IDs originate from "
        "Synthea patients.csv"
    )

    return index_df

# ============================================================
# FILTER EVENTS BEFORE INDEX
# ============================================================

def filter_before_index(
    df: pd.DataFrame,
    patient_column: str,
    date_column: str,
    index_df: pd.DataFrame,
):
    """
    Keep only events occurring before the member's index date.

    Strictly less than index_date is used.

    Therefore the target inpatient encounter itself cannot
    become a clinical predictor.
    """

    temp = df.copy()

    temp[patient_column] = (
        temp[patient_column]
        .astype(str)
    )

    temp = temp.merge(
        index_df[
            [
                "member_id",
                "index_date",
            ]
        ],
        left_on=patient_column,
        right_on="member_id",
        how="inner",
    )

    temp = temp[
        temp[date_column].notna()
        & temp["index_date"].notna()
        & (temp[date_column] < temp["index_date"])
    ].copy()

    return temp


# ============================================================
# ENCOUNTER FEATURES
# ============================================================

def build_encounter_features(
    encounters: pd.DataFrame,
    index_df: pd.DataFrame,
):
    """
    Build historical utilization features.
    """

    df = filter_before_index(
        encounters,
        "PATIENT",
        "START",
        index_df,
    )

    df["ENCOUNTERCLASS"] = (
        df["ENCOUNTERCLASS"]
        .astype(str)
        .str.lower()
    )

    features = (
        df.groupby("member_id")
        .agg(
            clinical_encounter_count=(
                "ENCOUNTERCLASS",
                "size",
            ),

            clinical_inpatient_history_count=(
                "ENCOUNTERCLASS",
                lambda x: (x == "inpatient").sum(),
            ),

            clinical_emergency_count=(
                "ENCOUNTERCLASS",
                lambda x: (x == "emergency").sum(),
            ),

            clinical_outpatient_count=(
                "ENCOUNTERCLASS",
                lambda x: (x == "outpatient").sum(),
            ),

            clinical_ambulatory_count=(
                "ENCOUNTERCLASS",
                lambda x: (x == "ambulatory").sum(),
            ),

            clinical_urgentcare_count=(
                "ENCOUNTERCLASS",
                lambda x: (x == "urgentcare").sum(),
            ),

            clinical_wellness_count=(
                "ENCOUNTERCLASS",
                lambda x: (x == "wellness").sum(),
            ),

            clinical_home_count=(
                "ENCOUNTERCLASS",
                lambda x: (x == "home").sum(),
            ),

            clinical_snf_count=(
                "ENCOUNTERCLASS",
                lambda x: (x == "snf").sum(),
            ),

            clinical_hospice_count=(
                "ENCOUNTERCLASS",
                lambda x: (x == "hospice").sum(),
            ),
        )
    )

    return features


# ============================================================
# CONDITION FEATURES
# ============================================================

def build_condition_features(
    conditions: pd.DataFrame,
    index_df: pd.DataFrame,
):
    """
    Build historical condition features.

    We use counts rather than hundreds of diagnosis-specific
    dummy variables because the dataset has only 108 members.
    """

    df = filter_before_index(
        conditions,
        "PATIENT",
        "START",
        index_df,
    )

    features = (
        df.groupby("member_id")
        .agg(
            clinical_condition_count=(
                "CODE",
                "size",
            ),

            clinical_unique_condition_count=(
                "CODE",
                "nunique",
            ),

            clinical_unique_condition_description_count=(
                "DESCRIPTION",
                "nunique",
            ),
        )
    )

    return features


# ============================================================
# OBSERVATION FEATURES
# ============================================================

def build_observation_features(
    observations: pd.DataFrame,
    index_df: pd.DataFrame,
):
    """
    Build historical observation features.
    """

    df = filter_before_index(
        observations,
        "PATIENT",
        "DATE",
        index_df,
    )

    features = (
        df.groupby("member_id")
        .agg(
            clinical_observation_count=(
                "CODE",
                "size",
            ),

            clinical_unique_observation_count=(
                "CODE",
                "nunique",
            ),
        )
    )

    return features


# ============================================================
# MEDICATION FEATURES
# ============================================================

def build_medication_features(
    medications: pd.DataFrame,
    index_df: pd.DataFrame,
):
    """
    Build historical medication features.
    """

    df = filter_before_index(
        medications,
        "PATIENT",
        "START",
        index_df,
    )

    features = (
        df.groupby("member_id")
        .agg(
            clinical_medication_count=(
                "CODE",
                "size",
            ),

            clinical_unique_medication_count=(
                "CODE",
                "nunique",
            ),
        )
    )

    return features


# ============================================================
# PROCEDURE FEATURES
# ============================================================

def build_procedure_features(
    procedures: pd.DataFrame,
    index_df: pd.DataFrame,
):
    """
    Build historical procedure features.
    """

    df = filter_before_index(
        procedures,
        "PATIENT",
        "START",
        index_df,
    )

    features = (
        df.groupby("member_id")
        .agg(
            clinical_procedure_count=(
                "CODE",
                "size",
            ),

            clinical_unique_procedure_count=(
                "CODE",
                "nunique",
            ),
        )
    )

    return features


# ============================================================
# ALLERGY FEATURES
# ============================================================

def build_allergy_features(
    allergies: pd.DataFrame,
    index_df: pd.DataFrame,
):
    """
    Build historical allergy features.

    Allergies only cover a subset of members in this synthetic
    dataset, so missing members are treated as zero observed
    allergies rather than dropping the member.
    """

    df = filter_before_index(
        allergies,
        "PATIENT",
        "START",
        index_df,
    )

    features = (
        df.groupby("member_id")
        .agg(
            clinical_allergy_count=(
                "CODE",
                "size",
            ),

            clinical_unique_allergy_count=(
                "CODE",
                "nunique",
            ),
        )
    )

    return features


# ============================================================
# IMMUNIZATION FEATURES
# ============================================================

def build_immunization_features(
    immunizations: pd.DataFrame,
    index_df: pd.DataFrame,
):
    """
    Build historical immunization features.
    """

    df = filter_before_index(
        immunizations,
        "PATIENT",
        "DATE",
        index_df,
    )

    features = (
        df.groupby("member_id")
        .agg(
            clinical_immunization_count=(
                "CODE",
                "size",
            ),

            clinical_unique_immunization_count=(
                "CODE",
                "nunique",
            ),
        )
    )

    return features


# ============================================================
# CAREPLAN FEATURES
# ============================================================

def build_careplan_features(
    careplans: pd.DataFrame,
    index_df: pd.DataFrame,
):
    """
    Build historical care-plan features.
    """

    df = filter_before_index(
        careplans,
        "PATIENT",
        "START",
        index_df,
    )

    features = (
        df.groupby("member_id")
        .agg(
            clinical_careplan_count=(
                "CODE",
                "size",
            ),

            clinical_unique_careplan_count=(
                "CODE",
                "nunique",
            ),
        )
    )

    return features


# ============================================================
# TIME FEATURES
# ============================================================

def build_time_features(
    index_df: pd.DataFrame,
):
    """
    Build observation-history duration features.

    These help distinguish a member with 2 encounters over
    one year from a member with 2 encounters over one month.
    """

    df = index_df.copy()

    df["history_start_date"] = pd.NaT

    # The earliest observed encounter is used as the beginning
    # of the member's clinical history.

    # This is populated by the caller.

    if "first_clinical_date" in df.columns:

        df["clinical_history_days"] = (
            df["index_date"]
            - df["first_clinical_date"]
        ).dt.days

    else:

        df["clinical_history_days"] = 0

    df["clinical_history_days"] = (
        df["clinical_history_days"]
        .fillna(0)
        .clip(lower=0)
    )

    return df[
        [
            "member_id",
            "clinical_history_days",
        ]
    ].set_index("member_id")


# ============================================================
# BUILD FEATURE CATALOG
# ============================================================

def build_feature_catalog():

    rows = [

        (
            "clinical_encounter_count",
            "Clinical utilization",
            "Number of encounters before index date",
        ),

        (
            "clinical_inpatient_history_count",
            "Clinical utilization",
            "Historical inpatient encounters before index date",
        ),

        (
            "clinical_emergency_count",
            "Clinical utilization",
            "Historical emergency encounters before index date",
        ),

        (
            "clinical_outpatient_count",
            "Clinical utilization",
            "Historical outpatient encounters before index date",
        ),

        (
            "clinical_ambulatory_count",
            "Clinical utilization",
            "Historical ambulatory encounters before index date",
        ),

        (
            "clinical_urgentcare_count",
            "Clinical utilization",
            "Historical urgent-care encounters before index date",
        ),

        (
            "clinical_wellness_count",
            "Clinical utilization",
            "Historical wellness encounters before index date",
        ),

        (
            "clinical_home_count",
            "Clinical utilization",
            "Historical home encounters before index date",
        ),

        (
            "clinical_snf_count",
            "Clinical utilization",
            "Historical skilled-nursing encounters before index date",
        ),

        (
            "clinical_hospice_count",
            "Clinical utilization",
            "Historical hospice encounters before index date",
        ),

        (
            "clinical_condition_count",
            "Clinical burden",
            "Number of historical condition records",
        ),

        (
            "clinical_unique_condition_count",
            "Clinical burden",
            "Number of unique historical condition codes",
        ),

        (
            "clinical_unique_condition_description_count",
            "Clinical burden",
            "Number of unique historical condition descriptions",
        ),

        (
            "clinical_observation_count",
            "Clinical monitoring",
            "Number of historical observations",
        ),

        (
            "clinical_unique_observation_count",
            "Clinical monitoring",
            "Number of unique observation codes",
        ),

        (
            "clinical_medication_count",
            "Medication burden",
            "Number of historical medication records",
        ),

        (
            "clinical_unique_medication_count",
            "Medication burden",
            "Number of unique historical medication codes",
        ),

        (
            "clinical_procedure_count",
            "Clinical utilization",
            "Number of historical procedures",
        ),

        (
            "clinical_unique_procedure_count",
            "Clinical utilization",
            "Number of unique historical procedure codes",
        ),

        (
            "clinical_allergy_count",
            "Clinical complexity",
            "Number of historical allergy records",
        ),

        (
            "clinical_unique_allergy_count",
            "Clinical complexity",
            "Number of unique historical allergy codes",
        ),

        (
            "clinical_immunization_count",
            "Preventive care",
            "Number of historical immunization records",
        ),

        (
            "clinical_unique_immunization_count",
            "Preventive care",
            "Number of unique historical immunization codes",
        ),

        (
            "clinical_careplan_count",
            "Care management",
            "Number of historical care plans",
        ),

        (
            "clinical_unique_careplan_count",
            "Care management",
            "Number of unique historical care-plan codes",
        ),

        (
            "clinical_history_days",
            "Temporal context",
            "Number of days of observed clinical history",
        ),
    ]

    catalog = pd.DataFrame(
        rows,
        columns=[
            "feature",
            "category",
            "description",
        ],
    )

    return catalog


# ============================================================
# BUILD MEMBER FEATURES
# ============================================================

def build_member_clinical_features():

    data = load_clinical_data()

    data = prepare_dates(
        data
    )

    patients = data["patients"]
    encounters = data["encounters"]

    # --------------------------------------------------------
    # Normalize member identifiers
    # --------------------------------------------------------

    patients = patients.copy()

    patients["Id"] = (
        patients["Id"]
        .astype(str)
    )

    encounters["PATIENT"] = (
        encounters["PATIENT"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Build target/index information
    # --------------------------------------------------------

    index_df = build_member_index_dates(
        patients,
        encounters,
    )

    # --------------------------------------------------------
    # Earliest clinical encounter
    # --------------------------------------------------------

    first_clinical = (
        encounters
        .groupby("PATIENT")["START"]
        .min()
        .rename("first_clinical_date")
    )

    index_df = index_df.merge(
        first_clinical,
        left_on="member_id",
        right_index=True,
        how="left",
    )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    time_features = build_time_features(
        index_df
    )

    # --------------------------------------------------------
    # Individual clinical feature groups
    # --------------------------------------------------------

    encounter_features = (
        build_encounter_features(
            encounters,
            index_df,
        )
    )

    condition_features = (
        build_condition_features(
            data["conditions"],
            index_df,
        )
    )

    observation_features = (
        build_observation_features(
            data["observations"],
            index_df,
        )
    )

    medication_features = (
        build_medication_features(
            data["medications"],
            index_df,
        )
    )

    procedure_features = (
        build_procedure_features(
            data["procedures"],
            index_df,
        )
    )

    allergy_features = (
        build_allergy_features(
            data["allergies"],
            index_df,
        )
    )

    immunization_features = (
        build_immunization_features(
            data["immunizations"],
            index_df,
        )
    )

    careplan_features = (
        build_careplan_features(
            data["careplans"],
            index_df,
        )
    )

    # --------------------------------------------------------
    # Combine all features
    # --------------------------------------------------------

    feature_table = index_df[
        [
            "patient_id",
            "member_id",
            "target_inpatient_any",
            "index_date",
        ]
    ].copy()

    feature_table = feature_table.set_index(
        "member_id"
    )

    feature_tables = [
        time_features,
        encounter_features,
        condition_features,
        observation_features,
        medication_features,
        procedure_features,
        allergy_features,
        immunization_features,
        careplan_features,
    ]

    for features in feature_tables:

        feature_table = feature_table.join(
            features,
            how="left",
        )

    # --------------------------------------------------------
    # Missing clinical history = zero
    # --------------------------------------------------------

    feature_columns = [
        col
        for col in feature_table.columns
        if col.startswith("clinical_")
    ]

    feature_table[feature_columns] = (
        feature_table[feature_columns]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Convert count-like features to numeric
    # --------------------------------------------------------

    for col in feature_columns:

        feature_table[col] = pd.to_numeric(
            feature_table[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Reset index
    # --------------------------------------------------------

    feature_table = (
        feature_table
        .reset_index()
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLINICAL FEATURE SUMMARY")
    print("=" * 70)

    print(
        f"Members: {len(feature_table)}"
    )

    print(
        "Positive target: "
        f"{int(feature_table['target_inpatient_any'].sum())}"
    )

    print(
        "Negative target: "
        f"{int((feature_table['target_inpatient_any'] == 0).sum())}"
    )

    print(
        f"Clinical features: {len(feature_columns)}"
    )

    print("\nFeature columns:")

    for col in feature_columns:
        print(
            f"  {col}"
        )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)

    forbidden = [
        "first_inpatient_date",
        "target_inpatient_any",
    ]

    leakage_features = [
        col
        for col in feature_columns
        if col in forbidden
    ]

    if leakage_features:

        raise ValueError(
            "Target leakage detected in clinical features: "
            f"{leakage_features}"
        )

    print(
        "Target variable is not included in "
        "clinical predictor features."
    )

    # --------------------------------------------------------
    # Check positive members
    # --------------------------------------------------------

    positive_count = (
        feature_table[
            "target_inpatient_any"
        ]
        .eq(1)
        .sum()
    )

    negative_count = (
        feature_table[
            "target_inpatient_any"
        ]
        .eq(0)
        .sum()
    )

    print(
        f"\nPositive members: {positive_count}"
    )

    print(
        f"Negative members: {negative_count}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_table.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    catalog = build_feature_catalog()

    catalog.to_csv(
        CATALOG_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(
        f"\nClinical features:\n{OUTPUT_FILE}"
    )

    print(
        f"\nClinical feature catalog:\n{CATALOG_FILE}"
    )

    return feature_table


# ============================================================
# MAIN
# ============================================================

def main():

    build_member_clinical_features()


if __name__ == "__main__":
    main()