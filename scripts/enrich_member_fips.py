from pathlib import Path

import pandas as pd

MEMBER_FILE = Path("data/interim/synthea_clean/members_1171.csv")
COUNTY_FILE = Path("data/processed/county_features.csv")
OUTPUT_FILE = Path("data/interim/synthea_clean/members_1171_geo.csv")


def normalize_county_name(series: pd.Series) -> pd.Series:
    """Normalize county names for deterministic matching."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+County$", "", regex=True)
        .str.strip()
        .str.lower()
    )


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

members = pd.read_csv(MEMBER_FILE)
counties = pd.read_csv(COUNTY_FILE)


# ---------------------------------------------------------
# Create matching keys
# ---------------------------------------------------------

members["county_lookup"] = normalize_county_name(
    members["county"]
)

members["state_lookup"] = (
    members["state"]
    .astype("string")
    .str.strip()
    .str.lower()
)

counties["county_lookup"] = normalize_county_name(
    counties["county_name"]
)

counties["state_lookup"] = (
    counties["state_name"]
    .astype("string")
    .str.strip()
    .str.lower()
)


# ---------------------------------------------------------
# Build county reference
# ---------------------------------------------------------

county_reference = (
    counties[
        [
            "county_lookup",
            "state_lookup",
            "county_fips",
        ]
    ]
    .drop_duplicates(
        subset=["county_lookup", "state_lookup"]
    )
)


# ---------------------------------------------------------
# Preserve original county_fips
# ---------------------------------------------------------

original_missing = members["county_fips"].isna()


# ---------------------------------------------------------
# Join county FIPS
# ---------------------------------------------------------

members = members.merge(
    county_reference,
    on=["county_lookup", "state_lookup"],
    how="left",
    validate="many_to_one",
    suffixes=("", "_mapped"),
)


# ---------------------------------------------------------
# Fill ONLY missing county_fips
# ---------------------------------------------------------

members.loc[
    members["county_fips"].isna(),
    "county_fips",
] = members.loc[
    members["county_fips"].isna(),
    "county_fips_mapped",
]


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

print("Total members:", len(members))

print(
    "Original missing county_fips:",
    int(original_missing.sum()),
)

print(
    "Final missing county_fips:",
    int(members["county_fips"].isna().sum()),
)

print("\nFIPS distribution:")
print(
    members["county_fips"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ---------------------------------------------------------
# Show unmatched counties
# ---------------------------------------------------------

unmatched = members[
    members["county_fips"].isna()
]

if not unmatched.empty:

    print("\nUNMATCHED COUNTIES:")

    print(
        unmatched[
            ["state", "county"]
        ]
        .drop_duplicates()
        .sort_values(["state", "county"])
        .to_string(index=False)
    )


# ---------------------------------------------------------
# Remove temporary columns
# ---------------------------------------------------------

members = members.drop(
    columns=[
        "county_lookup",
        "state_lookup",
        "county_fips_mapped",
    ]
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

members.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\nSaved:", OUTPUT_FILE)