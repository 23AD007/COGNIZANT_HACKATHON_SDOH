from sqlalchemy import text

from backend.data_store import initialize_development_database
from backend.database import get_engine


engine = get_engine()
initialize_development_database(engine)
with engine.connect() as connection:
    row = connection.execute(text("""
        SELECT COUNT(*) AS total, COUNT(lat) AS lat_present, COUNT(lon) AS lon_present
        FROM member_model_features
    """)).mappings().one()
    print(f"Total members: {row['total']}")
    print(f"Latitude present: {row['lat_present']}")
    print(f"Longitude present: {row['lon_present']}")
    print(f"Valid coordinates: {min(row['lat_present'], row['lon_present'])}")
