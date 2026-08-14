# Cognizant Hackathon SDOH

Reproducible, member-level social determinants of health (SDOH) risk-prioritization project.

## Current scope

This repository is organized for data preprocessing and feature engineering only. No machine-learning model is included at this stage.

## Data layout

- `data/raw/`: immutable source extracts (Synthea, ACS, CDC PLACES, USDA SRAM/Food Access Research Atlas, and Census data).
- `data/interim/`: reproducible intermediate outputs; not committed.
- `data/processed/`: validated analysis-ready outputs; not committed.

## Source layout

- `src/preprocessing/`: source-specific ingestion and validation.
- `src/features/`: future SDOH feature construction (not yet implemented).
- `src/geography/`: geographic key normalization and crosswalk utilities.
- `src/synthetic/`: Synthea-specific preparation.
- `src/modeling/`: reserved for a future modeling stage.
- `src/utils/`: shared helpers.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Raw datasets must remain unchanged. Derived data belongs only in `data/interim/` or `data/processed/`.

## Geographic integration status

Synthea currently provides member-level county FIPS where available, ZIP, city, state, county, and coordinates. The supplied ACS 2024 tables are national-level, while SRAM is tract-level. No legitimate Synthea member-to-tract crosswalk is present, so geographic joins are intentionally blocked.

Before an ACS-to-SRAM or member-to-tract join can be written, add: (1) legitimate tract-level 2024 ACS data with `GEO_ID` values in the `1400000US` tract form, and (2) a legitimate Synthea member-to-tract mapping/crosswalk.
