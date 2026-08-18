import pandas as pd

# Input files
patients_file = "data/raw/synthea/synthea_1171/csv/patients.csv"
county_file = "data/processed/county_features.csv"

# Load data
patients = pd.read_csv(patients_file)
counties = pd.read_csv(county_file)

# Normalize patient county/state names
patients["county_lookup"] = (
    patients["COUNTY"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+County$", "", regex=True)
    .str.strip()
)

patients["state_lookup"] = (
    patients["STATE"]
    .astype(str)
    .str.strip()
)

# Normalize reference county/state names
counties["county_lookup"] = (
    counties["county_name"]
    .astype(str)
    .str.strip()
)

counties["state_lookup"] = (
    counties["state_name"]
    .astype(str)
    .str.strip()
)

# Create county reference
county_reference = counties[
    [
        "county_lookup",
        "state_lookup",
        "county_fips",
        "county_name",
        "state_abbr",
    ]
].drop_duplicates(
    subset=["county_lookup", "state_lookup"]
)

# Map patients to county FIPS
patients_mapped = patients.merge(
    county_reference,
    on=["county_lookup", "state_lookup"],
    how="left",
    validate="many_to_one"
)

# Validation
print("Total patients:", len(patients_mapped))
print(
    "Patients with FIPS:",
    patients_mapped["county_fips"].notna().sum()
)
print(
    "Patients missing FIPS:",
    patients_mapped["county_fips"].isna().sum()
)

# Show unmatched counties
missing = patients_mapped[
    patients_mapped["county_fips"].isna()
]

if len(missing) > 0:
    print("\nCOUNTIES THAT COULD NOT BE MAPPED:")
    print(
        missing[
            ["STATE", "COUNTY"]
        ]
        .drop_duplicates()
        .sort_values(["STATE", "COUNTY"])
        .to_string(index=False)
    )

# Save
output_file = "data/interim/synthea_1171_with_fips.csv"

patients_mapped.to_csv(
    output_file,
    index=False
)

print("\nSaved:", output_file)