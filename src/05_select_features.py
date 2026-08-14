import pandas as pd

# Input master dataset
input_file = "data/processed/sram_2025_merged.csv"

# Output selected dataset
output_file = "data/processed/sram_selected_features.csv"

# Selected features
features = [
    # Geographic
    "CensusTract20",
    "State",
    "County20",
    "County24",

    # SRAM 1 - Socioeconomic / Demographic
    "Urban",
    "POP2020",
    "OHU2020",
    "GroupQuartersFlag",
    "NUMGQTRS",
    "PCTGQTRS",
    "LowIncomeTracts",
    "PovertyRate",
    "MedianFamilyIncome",
    "TractLOWI",
    "TractKids",
    "TractSeniors",
    "TractHUNV",
    "TractSNAP",
    "TractVeteran",
    "TractTribalArea",

    # SRAM 2 - Driving Distance
    "DD_SRAM_LILATracts_1And10",
    "DD_SRAM_LILATracts_halfAnd10",
    "DD_SRAM_LILATracts_1And20",
    "DD_SRAM_LILATracts_Vehicle",
    "DD_SRAM_HUNVFlag",
    "DD_SRAM_LAPOP1_10",
    "DD_SRAM_LAPOP05_10",
    "DD_SRAM_LAPOP1_20",
    "DD_SRAM_LALOWI1_10",
    "DD_SRAM_LALOWI05_10",
    "DD_SRAM_LALOWI1_20",

    # SRAM 3 - Straight Line Distance
    "SD_SRAM_LILATracts_1And10",
    "SD_SRAM_LILATracts_halfAnd10",
    "SD_SRAM_LILATracts_1And20",
    "SD_SRAM_LILATracts_Vehicle",
    "SD_SRAM_HUNVFlag",
    "SD_SRAM_LAPOP1_10",
    "SD_SRAM_LAPOP05_10",
    "SD_SRAM_LAPOP1_20",
    "SD_SRAM_LALOWI1_10",
    "SD_SRAM_LALOWI05_10",
    "SD_SRAM_LALOWI1_20"
]

# Read master dataset
df = pd.read_csv(input_file, low_memory=False)

print("Master dataset:")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Check missing features
missing = [col for col in features if col not in df.columns]

if missing:
    print("\nERROR: These features are missing:")
    for col in missing:
        print("-", col)
    exit()

# Select features
selected_df = df[features].copy()

# Save selected dataset
selected_df.to_csv(output_file, index=False)

print("\nSelected dataset created successfully!")
print("Rows:", len(selected_df))
print("Columns:", len(selected_df.columns))
print("Saved:", output_file)

print("\nSelected Features:")
for col in selected_df.columns:
    print("-", col)