"""Asynchronous, candidate-only retraining assessment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import json
import uuid

import pandas as pd

from .persistence import IngestionStore


class TrainingJobService:
    def __init__(self, store: IngestionStore, max_workers: int = 1) -> None:
        self.store, self.executor = store, ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, dataset_id: str) -> str:
        job_id = str(uuid.uuid4())
        self.store.create_job(job_id, dataset_id)
        self.executor.submit(self._evaluate, job_id, dataset_id)
        return job_id

    def _evaluate(self, job_id: str, dataset_id: str) -> None:
        self.store.update_job(job_id, "RUNNING")
        try:
            dataset = self.store.get_dataset(dataset_id)
            if dataset is None: raise ValueError("Unknown dataset.")
            report = dataset["report"]
            validation = report["validation"]
            if not validation.get("passed"):
                self.store.update_job(job_id, "REJECTED", result={"reason": "Dataset validation failed; no candidate model trained."})
                return
            # Uploaded values are metadata/observations until a dataset has an
            # explicit target and passes the existing model-training contract.
            schema_file = Path(__file__).resolve().parents[2] / "data" / "processed" / "models" / "member_risk_feature_schema.json"
            production = json.loads(schema_file.read_text(encoding="utf-8"))
            mapped = {entry["canonical_feature"] for entry in report["mappings"] if entry["canonical_feature"]}
            target = production["target_column"]
            coverage = len(set(production["feature_columns"]) & mapped) / len(production["feature_columns"])
            result = {"candidate_trained": False, "production_model": production["model_name"], "target_column": target, "feature_coverage": coverage, "promotion": "NOT_ELIGIBLE", "reason": "No explicit compatible training target was supplied; production artifacts were not modified."}
            self.store.update_job(job_id, "EVALUATED", result=result)
        except Exception as exc:
            self.store.update_job(job_id, "FAILED", error=str(exc))

    def promote(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(job_id)
        if job is None: raise KeyError(job_id)
        # Promotion is intentionally a state decision only. Candidate artifact
        # creation is not implemented until an explicit compatible target and
        # training path are supplied; production files remain untouched.
        if job["status"] != "EVALUATED" or job["result"].get("promotion") != "APPROVED":
            raise ValueError("Candidate model is not approved for promotion.")
        return job
