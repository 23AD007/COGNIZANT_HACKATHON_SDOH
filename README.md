# COGNIZANT_HACKATHON_SDOH

A Cognizant Hackathon project developed specifically to address the given business use case, leveraging AI/ML, data analytics, and intelligent decision-making to identify the problem, generate actionable insights, and recommend practical interventions.

## PLACES CSV Preprocessing

This repository contains `place.py`, a small preprocessing script for the CDC PLACES county-level CSV.

Usage

Run with defaults (input path set to the original dataset location):

```bash
python place.py
```

Specify input and output paths:

```bash
python place.py --input "D:\Downloads\dataset\dataset\PLACES__Local_Data_for_Better_Health,_County_Data,_2025_release_20260812.csv" --output cleaned_places.csv
```

Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

What the script does

- Loads the CSV (`low_memory=False` to avoid mixed-type warnings)
- Normalizes column names and trims whitespace
- Replaces common missing markers with `NaN`
- Attempts to coerce object columns to numeric where applicable
- Fills numeric NaNs with the column median
- Drops duplicates and writes a cleaned CSV

