from backend.database import engine
from sqlalchemy import text

with engine.connect() as conn:

    query = text("""
        SELECT
            member_id,
            age,
            gender,
            race,
            ethnicity,
            city,
            state,
            lat,
            lon
        FROM sdoh.member_sdoh_features
        LIMIT 10;
    """)

    result = conn.execute(query)

    print("\nMEMBER DATA")
    print("-" * 100)

    for row in result:
        print(
            f"ID: {row.member_id} | "
            f"Age: {row.age} | "
            f"Gender: {row.gender} | "
            f"Race: {row.race} | "
            f"City: {row.city} | "
            f"State: {row.state} | "
            f"Lat: {row.lat} | "
            f"Lon: {row.lon}"
        )