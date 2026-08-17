"""SQLAlchemy persistence for ingestion metadata; compatible with DATABASE_URL."""
from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Column, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.sql import func

from backend.database import Base, get_engine

from .external_ingestion import IngestionResult


class IngestedDataset(Base):
    __tablename__ = "ingested_datasets"
    dataset_id = Column(String(36), primary_key=True)
    source_name = Column(String(255), nullable=False)
    source_format = Column(String(16), nullable=False)
    status = Column(String(32), nullable=False)
    report = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class TrainingJobRecord(Base):
    __tablename__ = "training_jobs"
    job_id = Column(String(36), primary_key=True)
    dataset_id = Column(String(36), nullable=False, index=True)
    status = Column(String(32), nullable=False)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class IngestionStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.engine = get_engine(database_url)
        Base.metadata.create_all(self.engine)

    def save_ingestion(self, result: IngestionResult) -> str:
        with Session(self.engine) as session:
            session.merge(IngestedDataset(dataset_id=result.dataset_id, source_name=result.source_name, source_format=result.source_format, status="VALIDATED" if result.validation["passed"] else "INVALID", report=result.report()))
            session.commit()
        return result.dataset_id

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            row = session.get(IngestedDataset, dataset_id)
            return None if row is None else {"dataset_id": row.dataset_id, "status": row.status, "report": row.report}

    def create_job(self, job_id: str, dataset_id: str) -> None:
        with Session(self.engine) as session:
            session.add(TrainingJobRecord(job_id=job_id, dataset_id=dataset_id, status="QUEUED"))
            session.commit()

    def update_job(self, job_id: str, status: str, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        with Session(self.engine) as session:
            row = session.get(TrainingJobRecord, job_id)
            if row is None: raise KeyError(job_id)
            row.status, row.result, row.error = status, result, error
            session.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            row = session.get(TrainingJobRecord, job_id)
            return None if row is None else {"job_id": row.job_id, "dataset_id": row.dataset_id, "status": row.status, "result": row.result, "error": row.error}
