from sqlalchemy import text
from backend.database import engine

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT
            COUNT(*) AS total,
            COUNT(lat) AS lat_present,
            COUNT(lon) AS lon_present,
            COUNT(*) FILTER (
                WHERE lat IS NOT NULL
                  AND lon IS NOT NULL
            ) AS valid_coordinates
        FROM sdoh.member_sdoh_features
    """)).fetchone()

    print("Total members:", result.total)
    print("Latitude present:", result.lat_present)
    print("Longitude present:", result.lon_present)
    print("Valid coordinates:", result.valid_coordinates)