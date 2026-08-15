from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def validate_member_table(df: pd.DataFrame) -> None:
    required = {"member_id", "county_fips"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Member table is missing required columns: {sorted(missing)}"
        )

    if df["member_id"].isna().any():
        raise ValueError("Member table contains missing member_id values.")

    if df["member_id"].duplicated().any():
        raise ValueError("Member table contains duplicate member_id values.")

    county = df["county_fips"].astype("string").str.strip()

    # Missing county_fips is allowed.
    # These members will remain in the output but cannot receive
    # county-level SDOH features without a legitimate county mapping.
    non_missing = county.notna() & county.ne("")

    if not county[non_missing].str.fullmatch(
        r"\d{5}",
        na=False,
    ).all():
        raise ValueError(
            "Member table contains invalid non-missing county_fips values."
        )

def validate_county_sdoh_table(df: pd.DataFrame) -> None:
    if "county_fips" not in df.columns:
        raise ValueError(
            "County SDOH table is missing county_fips."
        )

    county = df["county_fips"].astype("string")

    if county.isna().any() or county.eq("").any():
        raise ValueError(
            "County SDOH table contains missing county_fips."
        )

    if not county.str.fullmatch(r"\d{5}", na=False).all():
        raise ValueError(
            "County SDOH table contains invalid county_fips."
        )

    if county.duplicated().any():
        raise ValueError(
            "County SDOH table contains duplicate county_fips."
        )


def enrich_members(
    members_path: str | Path,
    county_sdoh_path: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    members_path = Path(members_path)
    county_sdoh_path = Path(county_sdoh_path)
    output_path = Path(output_path)

    members = pd.read_csv(
        members_path,
        dtype={"member_id": "string", "county_fips": "string"},
    )

    county_sdoh = pd.read_csv(
        county_sdoh_path,
        dtype={"county_fips": "string"},
    )

    validate_member_table(members)
    validate_county_sdoh_table(county_sdoh)

    members["county_fips"] = (
    members["county_fips"]
    .astype("string")
    .str.strip()
    )
    county_sdoh["county_fips"] = county_sdoh["county_fips"].str.strip()

    member_count_before = len(members)

    member_counties = set(
        members.loc[
            members["county_fips"].notna()
            & members["county_fips"].ne(""),
            "county_fips",
        ]
    )

    matched_counties = member_counties.intersection(
        set(county_sdoh["county_fips"])
    )

    enriched = members.merge(
        county_sdoh,
        on="county_fips",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if len(enriched) != member_count_before:
        raise RuntimeError(
            "Member row count changed during enrichment."
        )

    matched_members = int(
        enriched["_merge"].eq("both").sum()
    )

    unmatched_members = int(
        enriched["_merge"].eq("left_only").sum()
    )

    enriched = enriched.drop(columns=["_merge"])

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_csv(
        output_path,
        index=False,
    )

    return {
        "members_before": member_count_before,
        "members_after": len(enriched),
        "matched_members": matched_members,
        "unmatched_members": unmatched_members,
        "matched_counties": len(matched_counties),
        "county_sdoh_count": len(county_sdoh),
        "final_columns": len(enriched.columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach county-level SDOH features to Synthea members."
    )

    parser.add_argument(
        "--members",
        type=Path,
        default=Path(
            "data/interim/synthea_clean/members.csv"
        ),
    )

    parser.add_argument(
        "--county-sdoh",
        type=Path,
        default=Path(
            "data/processed/county_sdoh_features.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/member_sdoh_features.csv"
        ),
    )

    args = parser.parse_args()

    summary = enrich_members(
        members_path=args.members,
        county_sdoh_path=args.county_sdoh,
        output_path=args.output,
    )

    print("Member SDOH enrichment complete.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()