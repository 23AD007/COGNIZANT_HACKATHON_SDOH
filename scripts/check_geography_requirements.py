"""Read-only availability check for required tract-level geographic inputs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    from src.preprocessing.geography import load_member_to_tract_crosswalk
    from src.preprocessing.geography_validation import TRACT_ACS_GEOID_PATTERN
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.preprocessing.geography import load_member_to_tract_crosswalk
    from src.preprocessing.geography_validation import TRACT_ACS_GEOID_PATTERN


def tract_acs_available(acs_dir: Path) -> bool:
    for path in acs_dir.glob("*-Data.csv"):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "GEO_ID" not in (reader.fieldnames or []):
                continue
            if any(TRACT_ACS_GEOID_PATTERN.fullmatch((row.get("GEO_ID") or "").strip()) for row in reader):
                return True
    return False


def member_tract_crosswalk_available(crosswalk_path: Path | None) -> bool:
    try:
        load_member_to_tract_crosswalk(crosswalk_path)
    except (FileNotFoundError, ValueError):
        return False
    return True


def sram_tract_data_available(sram_dir: Path) -> bool:
    for path in sram_dir.glob("*Distance Data.csv"):
        with path.open(encoding="latin-1", newline="") as handle:
            header = next(csv.reader(handle), [])
        if "CensusTract20" in header:
            return True
    return False


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Report whether required tract-level geographic inputs are available.")
    parser.add_argument("--acs-dir", type=Path, default=Path("data/raw/acs"))
    parser.add_argument("--sram-dir", type=Path, default=Path("data/raw/sram"))
    parser.add_argument("--member-tract-crosswalk", type=Path, default=None)
    args = parser.parse_args()
    print(f"TRACT ACS AVAILABLE: {'YES' if tract_acs_available(args.acs_dir) else 'NO'}")
    print(f"MEMBER→TRACT CROSSWALK AVAILABLE: {'YES' if member_tract_crosswalk_available(args.member_tract_crosswalk) else 'NO'}")
    print(f"SRAM TRACT DATA AVAILABLE: {'YES' if sram_tract_data_available(args.sram_dir) else 'NO'}")


if __name__ == "__main__":
    main()
