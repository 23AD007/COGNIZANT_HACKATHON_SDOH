from pathlib import Path

from src.preprocessing.acs import preprocess_acs
from src.preprocessing.sram import preprocess_sram


ROOT = Path(__file__).resolve().parents[1]


ACS_INPUT = (
    ROOT
    / "data"
    / "raw"
    / "acs"
    / "acs_reference.csv"
)

ACS_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "acs"
    / "acs_features.csv"
)


SRAM_INPUT = (
    ROOT
    / "data"
    / "raw"
    / "sram"
    / "food_access.csv"
)

SRAM_OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "sram"
    / "food_access_features.csv"
)


if __name__ == "__main__":

    preprocess_acs(
        ACS_INPUT,
        ACS_OUTPUT
    )

    preprocess_sram(
        SRAM_INPUT,
        SRAM_OUTPUT
    )