from sqlalchemy import text

from backend.data_store import initialize_development_database
from backend.database import get_engine


engine = get_engine()
initialize_development_database(engine)
with engine.connect() as connection:
    rows = connection.execute(text("""
        SELECT member_id, age, gender, race, ethnicity, city, state, lat, lon
        FROM member_model_features ORDER BY member_id LIMIT 10
    """)).mappings()
    for row in rows:
        print(" | ".join(f"{key}: {value}" for key, value in row.items()))
