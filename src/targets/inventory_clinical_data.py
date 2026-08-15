from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SYNTHEA_DIR = ROOT / "data" / "raw" / "synthea"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_file(path: Path) -> dict:
    try:
        df = pd.read_csv(path, low_memory=False)

        return {
            "file": path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "has_patient_id": "PATIENT" in df.columns,
            "has_start": "START" in df.columns,
            "has_stop": "STOP" in df.columns,
            "has_code": "CODE" in df.columns,
            "has_description": "DESCRIPTION" in df.columns,
            "columns_list": "|".join(df.columns.astype(str)),
            "status": "readable",
        }

    except Exception as exc:
        return {
            "file": path.name,
            "rows": None,
            "columns": None,
            "has_patient_id": False,
            "has_start": False,
            "has_stop": False,
            "has_code": False,
            "has_description": False,
            "columns_list": "",
            "status": f"ERROR: {exc}",
        }


def main():
    if not SYNTHEA_DIR.exists():
        raise FileNotFoundError(
            f"Synthea directory not found: {SYNTHEA_DIR}"
        )

    csv_files = sorted(SYNTHEA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {SYNTHEA_DIR}"
        )

    results = [inspect_file(path) for path in csv_files]

    report = pd.DataFrame(results)

    output = OUTPUT_DIR / "clinical_data_inventory.csv"
    report.to_csv(output, index=False)

    print("=" * 60)
    print("SYNTHEA CLINICAL DATA INVENTORY")
    print("=" * 60)
    print(f"Directory: {SYNTHEA_DIR}")
    print(f"CSV files: {len(csv_files)}")
    print()
    print(report[
        [
            "file",
            "rows",
            "columns",
            "has_patient_id",
            "has_start",
            "has_stop",
            "has_code",
            "has_description",
            "status",
        ]
    ].to_string(index=False))

    print()
    print(f"Report written to: {output}")


if __name__ == "__main__":
    main()