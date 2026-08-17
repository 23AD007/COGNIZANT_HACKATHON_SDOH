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
