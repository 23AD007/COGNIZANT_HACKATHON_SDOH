from sqlalchemy import text
from sqlalchemy.orm import Session


def get_all_members(db: Session):

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
        ORDER BY member_id
    """)

    result = db.execute(query)

    return [
        {
            "member_id": row.member_id,
            "age": row.age,
            "gender": row.gender,
            "race": row.race,
            "ethnicity": row.ethnicity,
            "city": row.city,
            "state": row.state,
            "latitude": row.lat,
            "longitude": row.lon,
        }
        for row in result
    ]