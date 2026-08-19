from pathlib import Path

import pandas as pd

from scripts.check_geography_readiness import (
    sram_tract_status,
)


PROCESSED_DIR = Path("data/processed")
MEMBER_ARTIFACTS = (
    "member_model_features.csv",
    "member_sdoh_features.csv",
    "member_sdoh_model_features.csv",
)


def test_current_member_geography_artifacts_are_complete_and_county_backed():
    member_frames = [
        pd.read_csv(PROCESSED_DIR / artifact, usecols=["member_id", "county_fips"], dtype="string")
        for artifact in MEMBER_ARTIFACTS
    ]
    member_ids = [set(frame["member_id"]) for frame in member_frames]

    assert all(len(frame) == 1279 for frame in member_frames)
    assert all(frame["member_id"].notna().all() and frame["member_id"].is_unique for frame in member_frames)
    assert member_ids[0] == member_ids[1] == member_ids[2]
    assert all(frame["county_fips"].str.fullmatch(r"\d{5}", na=False).all() for frame in member_frames)

    county_features = pd.read_csv(PROCESSED_DIR / "county_features.csv", usecols=["county_fips"], dtype="string")
    valid_county_fips = set(county_features.loc[county_features["county_fips"].str.fullmatch(r"\d{5}", na=False), "county_fips"])
    assert set(member_frames[1]["county_fips"]).issubset(valid_county_fips)


def test_current_tract_geography_artifact_is_valid():
    available, rows, missing_keys, _ = sram_tract_status(Path("data/processed/sram_features.csv"))
    assert available
    assert rows > 0
    assert missing_keys == 0
