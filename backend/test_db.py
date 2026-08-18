from database import engine
from sqlalchemy import text


try:
    with engine.connect() as connection:

        result = connection.execute(
            text("""
                SELECT
                    m.member_id,
                    m.city,
                    m.state,

                    c.geoid,
                    c.name AS county_name,

                    s.unemployment_pct,
                    s.uninsured_pct,
                    s.public_assistance_pct

                FROM sdoh.member_sdoh_features AS m

                JOIN sdoh.county_boundaries AS c
                    ON ST_Within(m.geom, c.geom)

                LEFT JOIN sdoh.county_sdoh_features AS s
                    ON s.county_fips = c.geoid::bigint

                WHERE m.geom IS NOT NULL

                LIMIT 10;
            """)
        )

        print("Member → County → SDOH successful!\n")

        for row in result:
            print(row)

except Exception as e:
    print("Query failed!")
    print(e)