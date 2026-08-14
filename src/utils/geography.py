import pandas as pd


def normalize_fips(df, column="fips"):

    df = df.copy()

    df[column] = (
        df[column]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(5)
    )

    return df


def attach_geographic_sdoh(
    members,
    acs,
    places,
    food_access
):

    members = normalize_fips(
        members,
        "fips"
    )

    acs = normalize_fips(
        acs,
        "fips"
    )

    places = normalize_fips(
        places,
        "fips"
    )

    food_access = normalize_fips(
        food_access,
        "fips"
    )

    members = members.merge(
        acs,
        on="fips",
        how="left",
        suffixes=("", "_acs")
    )

    members = members.merge(
        places,
        on="fips",
        how="left",
        suffixes=("", "_places")
    )

    members = members.merge(
        food_access,
        on="fips",
        how="left",
        suffixes=("", "_food")
    )

    return members
        out = out.rename(columns={"FIPS": "county_fips"})

    if out.empty:
        return out

    # ACS: county-level rows keyed by the 5-digit county FIPS embedded in GEO_ID.
    if acs is not None and not acs.empty:
        acs_df = acs.copy()
        if "GEO_ID" in acs_df.columns:
            acs_df["county_fips"] = (
                acs_df["GEO_ID"]
                .astype(str)
                .str.extract(r"0500000US(\d{5})", expand=False)
                .str.zfill(5)
            )
            acs_df = normalize_fips(acs_df, "county_fips")
            acs_features = acs_df.drop_duplicates(subset=["county_fips"]).copy()
            acs_features = acs_features.drop(columns=["county_fips"], errors="ignore")
            out = out.merge(
                acs_df[[c for c in acs_df.columns if c == "county_fips" or c in acs_df.columns]].drop_duplicates(subset=["county_fips"]),
                on="county_fips",
                how="left",
                suffixes=("", "_acs"),
            )

    # PLACES: county-level rows keyed by state + county name after aggregating repeated measures.
    if places is not None and not places.empty:
        places_df = places.copy()
        if "StateDesc" in places_df.columns:
            places_df = places_df.rename(columns={"StateDesc": "State"})
        if "LocationName" in places_df.columns:
            places_df = _state_county_key(places_df, "State", "LocationName")
            geo_key = places_df[["__state_key", "__county_key"]].copy()
            numeric_cols = [
                col for col in ["Data_Value", "Low_Confidence_Limit", "High_Confidence_Limit", "TotalPopulation"]
                if col in places_df.columns
            ]
            if numeric_cols:
                places_numeric = (
                    places_df[["__state_key", "__county_key", *numeric_cols]]
                    .groupby(["__state_key", "__county_key"], dropna=False)[numeric_cols]
                    .mean()
                    .reset_index()
                )
                places_numeric["__geo_key"] = places_numeric["__state_key"] + "|" + places_numeric["__county_key"]
                places_summary = places_numeric.drop(columns=["__state_key", "__county_key"], errors="ignore")
                out = out.assign(__geo_key=out["state"].astype(str).str.strip().str.lower() + "|" + out["county"].astype(str).map(_county_name_key).fillna(""))
                out = out.merge(places_summary, on="__geo_key", how="left", suffixes=("", "_places"))
                out = out.drop(columns=["__geo_key"], errors="ignore")

    # Food access: aggregate tract-level data to county FIPS before merging.
    if food_access is not None and not food_access.empty:
        food_df = food_access.copy()
        tract_col = "CensusTract20" if "CensusTract20" in food_df.columns else "CensusTract" if "CensusTract" in food_df.columns else None
        if tract_col is not None:
            food_df["county_fips"] = (
                food_df[tract_col]
                .astype(str)
                .str.replace(".0", "", regex=False)
                .str.slice(0, 5)
                .str.zfill(5)
            )
            numeric_cols = [
                col for col in food_df.select_dtypes(include="number").columns
                if col not in {"county_fips"}
            ]
            if numeric_cols:
                food_summary = (
                    food_df[["county_fips", *numeric_cols]]
                    .groupby("county_fips", dropna=False)[numeric_cols]
                    .mean()
                    .reset_index()
                )
                out = out.merge(food_summary, on="county_fips", how="left", suffixes=("", "_food"))

    return out