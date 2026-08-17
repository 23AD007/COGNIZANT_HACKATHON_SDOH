import json
import io

import pandas as pd

from src.ingestion.canonical_schema import CanonicalSchema
from src.ingestion.external_ingestion import ingest_file
from src.ingestion.persistence import IngestionStore
from src.ingestion.semantic_mapping import SemanticMapper
from src.ingestion.training_jobs import TrainingJobService
from backend.app import create_app
from backend.database import database_url


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


def test_flask_upload_and_training_job_lifecycle(tmp_path):
    store = IngestionStore(f"sqlite:///{tmp_path / 'api.sqlite'}")
    client = create_app(store).test_client()
    response = client.post(
        "/datasets",
        data={"file": (io.BytesIO(b"member_id,age\nmember-1,42\n"), "members.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    dataset_id = response.get_json()["dataset_id"]
    job_response = client.post(f"/datasets/{dataset_id}/training-jobs")
    assert job_response.status_code == 202
    assert client.get(f"/training-jobs/{job_response.get_json()['job_id']}").status_code == 200


def test_explicit_development_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FLASK_ENV", "development")
    assert database_url().startswith("sqlite:///")
