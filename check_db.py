from sqlalchemy import text

from backend.data_store import ARTIFACT_TABLES, initialize_development_database
from backend.database import get_engine


engine = get_engine()
initialize_development_database(engine)
with engine.connect() as connection:
    for table_name in (*ARTIFACT_TABLES, "member_clinical_features", "member_recommendations"):
        count = connection.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar_one()
        print(f"{table_name}: {count}")
