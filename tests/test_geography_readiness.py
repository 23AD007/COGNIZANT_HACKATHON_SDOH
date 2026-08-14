from pathlib import Path

from scripts.check_geography_readiness import (
    member_tract_crosswalk_available,
    sram_tract_status,
    synthea_member_geography_available,
    tract_acs_available,
)


def test_current_repository_geography_availability():
    assert synthea_member_geography_available(Path("data/raw/synthea/patients.csv"))
    assert not member_tract_crosswalk_available(None)
    assert not tract_acs_available(Path("data/raw/acs"))
    available, rows, missing_keys, _ = sram_tract_status(Path("data/processed/sram_features.csv"))
    assert available
    assert rows > 0
    assert missing_keys == 0
