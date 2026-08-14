from pathlib import Path
import re
import pandas as pd


ACS_TABLES = [
    "DP02",
    "DP03",
    "DP04",
    "DP05",
    "S0101",
    "S1501",
    "S1701",
    "S1901",
]


def find_data_files(input_dir: Path):
    """
    Find actual ACS data CSVs.

    Excludes:
      - Column-Metadata files
      - metadata files
      - table notes
      - previously generated clean files
    """

    files = sorted(input_dir.glob("*.csv"))

    data_files = []

    for file_path in files:

        name = file_path.name.lower()

        # Explicitly exclude metadata.
        if "column-metadata" in name:
            continue

        if "metadata" in name:
            continue

        if "table-notes" in name:
            continue

        # Ignore our own outputs if they happen to be
        # placed in the raw directory.
        if "_clean" in name:
            continue

        data_files.append(file_path)

    return data_files


def identify_table(file_path: Path):

    name = file_path.name.upper()

    for table in ACS_TABLES:

        if re.search(
            rf"(?<![A-Z0-9]){table}(?![A-Z0-9])",
            name
        ):
            return table

    return None


def find_geo_columns(df):

    upper_to_original = {
        str(c).upper(): c
        for c in df.columns
    }

    geo_id = upper_to_original.get("GEO_ID")
    name = upper_to_original.get("NAME")

    return geo_id, name


def clean_geo_id(series):

    return (
        series
        .astype("string")
        .str.strip()
    )


def remove_moe_columns(df):

    keep = []

    for col in df.columns:

        upper = str(col).upper()

        if col in ["GEO_ID", "NAME"]:
            keep.append(col)
            continue

        # Census MOE columns commonly end in M.
        if upper.endswith("M"):
            continue

        keep.append(col)

    return df[keep].copy()


def convert_numeric_columns(df):

    protected = {
        "GEO_ID",
        "NAME",
    }

    for col in df.columns:

        if col in protected:
            continue

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        # Only replace when conversion gives
        # meaningful numeric values.
        if converted.notna().sum() > 0:

            df[col] = converted

    return df


def preprocess_acs(
    input_dir="data/raw/acs",
    output_dir="data/interim/acs_clean",
):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():

        raise FileNotFoundError(
            f"ACS directory not found: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    data_files = find_data_files(
        input_dir
    )

    if not data_files:

        raise FileNotFoundError(
            f"No ACS data CSVs found in {input_dir}"
        )

    manifest = []

    print("\nACS files detected:\n")

    for file_path in data_files:

        table = identify_table(
            file_path
        )

        print(
            f"{file_path.name} "
            f"→ {table}"
        )

        if table is None:

            print(
                f"WARNING: Could not identify table "
                f"for {file_path.name}"
            )

            continue

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        print(
            f"  raw shape = {df.shape}"
        )

        geo_id, name = find_geo_columns(
            df
        )

        if geo_id is None:

            print(
                f"  SKIP: GEO_ID not found"
            )

            continue

        df[geo_id] = clean_geo_id(
            df[geo_id]
        )

        if name is not None:

            df[name] = (
                df[name]
                .astype("string")
                .str.strip()
            )

        # Remove MOE columns.
        df = remove_moe_columns(
            df
        )

        # Convert numeric fields.
        df = convert_numeric_columns(
            df
        )

        # Remove completely empty rows.
        df = df.dropna(
            how="all"
        )

        # Validate.
        if len(df) <= 2:

            raise ValueError(
                f"{file_path.name} produced only "
                f"{len(df)} data rows. "
                f"This is too small for the expected "
                f"geographic ACS dataset. "
                f"Inspect the raw file before continuing."
            )

        if df[geo_id].isna().any():

            raise ValueError(
                f"{file_path.name} contains missing "
                f"GEO_ID values."
            )

        if df[geo_id].duplicated().any():

            duplicate_count = (
                df[geo_id]
                .duplicated()
                .sum()
            )

            print(
                f"  WARNING: "
                f"{duplicate_count} duplicate GEO_IDs"
            )

        output_path = (
            output_dir
            / f"{table.lower()}_clean.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )

        manifest.append(
            {
                "table": table,
                "input_file": file_path.name,
                "output_file": output_path.name,
                "rows": len(df),
                "columns": len(df.columns),
                "geo_column": geo_id,
            }
        )

        print(
            f"  cleaned shape = {df.shape}"
        )

    if not manifest:

        raise RuntimeError(
            "No ACS datasets were successfully processed."
        )

    manifest_df = pd.DataFrame(
        manifest
    )

    manifest_df.to_csv(
        output_dir / "metadata_manifest.csv",
        index=False
    )

    print(
        "\nACS preprocessing completed."
    )

    print(
        manifest_df.to_string(
            index=False
        )
    )

    return manifest_df


if __name__ == "__main__":

    preprocess_acs()