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


def find_acs_data_files(input_dir: Path):
    """Return only ACS *-Data.csv files."""

    files = sorted(
        input_dir.glob("*-Data.csv")
    )

    return [
        f for f in files
        if "Column-Metadata" not in f.name
    ]


def identify_table(file_path: Path):

    name = file_path.name.upper()

    for table in ACS_TABLES:

        if f".{table}-DATA.CSV" in name:
            return table

    return None


def clean_geo_id(series):

    return (
        series
        .astype("string")
        .str.strip()
    )


def select_estimate_columns(df):

    keep = []

    for col in df.columns:

        col = str(col)

        # Geographic identifiers
        if col in {"GEO_ID", "NAME"}:
            keep.append(col)
            continue

        # ACS estimates
        if re.search(r"_\d+E$", col):
            keep.append(col)
            continue

        # Decennial P1/P2 estimates have N suffix.
        if re.search(r"_\d+N$", col):
            keep.append(col)
            continue

    return df[keep].copy()


def convert_numeric_columns(df):

    protected = {
        "GEO_ID",
        "NAME",
    }

    for col in df.columns:

        if col in protected:
            continue

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

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

    files = find_acs_data_files(
        input_dir
    )

    if not files:
        raise FileNotFoundError(
            "No *-Data.csv ACS files found."
        )

    manifest = []

    for file_path in files:

        table = identify_table(
            file_path
        )

        # Ignore Decennial here.
        if table is None:
            continue

        print(f"\nProcessing {table}")
        print(f"Input: {file_path.name}")

        df = pd.read_csv(
            file_path,
            low_memory=False
        )

        print(
            f"Raw shape: {df.shape}"
        )

        required = {
            "GEO_ID",
            "NAME",
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                f"{file_path.name} is missing "
                f"{missing}"
            )

        df["GEO_ID"] = clean_geo_id(
            df["GEO_ID"]
        )

        df["NAME"] = (
            df["NAME"]
            .astype("string")
            .str.strip()
        )

        # Keep estimates only.
        df = select_estimate_columns(
            df
        )

        df = convert_numeric_columns(
            df
        )

        df = df.dropna(
            how="all",
            subset=[
                c for c in df.columns
                if c not in {"GEO_ID", "NAME"}
            ]
        )

        if len(df) == 0:
            raise ValueError(
                f"{table} has zero usable rows."
            )

        if df["GEO_ID"].isna().any():
            raise ValueError(
                f"{table} contains missing GEO_ID."
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
                "geo_column": "GEO_ID",
            }
        )

        print(
            f"Clean shape: {df.shape}"
        )

        print(
            f"Saved: {output_path}"
        )

    manifest_df = pd.DataFrame(
        manifest
    )

    manifest_df.to_csv(
        output_dir / "metadata_manifest.csv",
        index=False
    )

    print(
        "\nACS preprocessing complete."
    )

    print(
        manifest_df.to_string(
            index=False
        )
    )

    return manifest_df


if __name__ == "__main__":
    preprocess_acs()