from pathlib import Path

from src.preprocessing.synthea_patients import (
    preprocess_patients
)


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "data"
    / "raw"
    / "synthea"
    / "patients.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "synthea"
    / "patients_preprocessed.csv"
)


if __name__ == "__main__":

    preprocess_patients(
        input_path=INPUT,
        output_path=OUTPUT
    )