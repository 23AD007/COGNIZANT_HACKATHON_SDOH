from sqlalchemy import text
from backend.database import engine

queries = {
    "patients": "SELECT COUNT(*) FROM public.patients",
    "conditions": "SELECT COUNT(*) FROM public.conditions",
    "observations": "SELECT COUNT(*) FROM public.observations",
    "member_sdoH": "SELECT COUNT(*) FROM sdoh.member_sdoh_features",
    "county_sdoh": "SELECT COUNT(*) FROM sdoh.county_sdoh_features",
    "clinical_coverage": "SELECT COUNT(*) FROM sdoh.clinical_patient_coverage",
}

with engine.connect() as conn:
    for name, query in queries.items():
        count = conn.execute(text(query)).scalar()
        print(f"{name}: {count}")