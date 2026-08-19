import pandas as pd

from backend.database import database_url
from backend.main import app, health
from src.ingestion.canonical_schema import CanonicalSchema
from src.ingestion.external_ingestion import ingest_file
from src.ingestion.persistence import IngestionStore
from src.ingestion.semantic_mapping import SemanticMapper
from src.ingestion.training_jobs import TrainingJobService


def schema():
    return CanonicalSchema(member_id_column="member_id", model_features=["age", "clinical_condition_count"])


def test_mapping_statuses():
    mapper = SemanticMapper(schema(), {"years": "age"})
    assert mapper.map_field("age").status == "EXACT"
    assert mapper.map_field("Clinical Condition Count").status == "NORMALIZED"
    assert mapper.map_field("years").status == "ALIAS"
    assert mapper.map_field("unknown").status == "UNMAPPED"


def test_partial_tabular_ingestion_preserves_unknown_columns(tmp_path):
    source = tmp_path / "members.csv"
    pd.DataFrame({"member_id": ["member-1"], "years": [42], "unmapped": ["kept in report"]}).to_csv(source, index=False)
    result = ingest_file(source, schema(), aliases={"years": "age"})
    assert result.validation["passed"]
    assert result.validation["unmapped_fields"] == ["unmapped"]
    assert result.records[0].provenance["canonical_features"] == {"age": 42}


def test_document_observations_include_provenance(tmp_path):
    source = tmp_path / "note.txt"
    source.write_text("member note", encoding="utf-8")
    result = ingest_file(source, schema())
    assert result.observations[0]["source_document"] == "note.txt"
    assert result.observations[0]["mapping_status"] == "UNMAPPED"


def test_persistence_and_safe_candidate_job(tmp_path):
    source = tmp_path / "members.csv"
    pd.DataFrame({"member_id": ["member-1"], "age": [42]}).to_csv(source, index=False)
    result = ingest_file(source, schema())
    store = IngestionStore(f"sqlite:///{tmp_path / 'metadata.sqlite'}")
    store.save_ingestion(result)
    service = TrainingJobService(store)
    job_id = service.submit(result.dataset_id)
    service.executor.shutdown(wait=True)
    job = store.get_job(job_id)
    assert job["status"] == "EVALUATED"
    assert job["result"]["candidate_trained"] is False
    assert job["result"]["promotion"] == "NOT_ELIGIBLE"


def test_fastapi_health_endpoint_uses_current_entry_point():
    health_route = next(route for route in app.routes if getattr(route, "path", None) == "/health")
    assert "GET" in health_route.methods
    assert health() == {"status": "healthy"}


def test_explicit_development_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APP_ENV", "development")
    assert database_url().startswith("sqlite:///")
