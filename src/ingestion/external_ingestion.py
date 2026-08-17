"""File extraction, conservative canonicalization, and provenance reporting."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping
import json
import uuid

import pandas as pd

from .canonical_schema import CanonicalMemberRecord, CanonicalSchema, _is_missing
from .semantic_mapping import FieldMapping, SemanticMapper
from .validation import validate_canonical_members


SUPPORTED_TABULAR_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}
SUPPORTED_DOCUMENT_FORMATS = {".txt", ".pdf", ".docx"}


@dataclass
class IngestionResult:
    dataset_id: str
    source_name: str
    source_format: str
    schema: list[str]
    mappings: list[FieldMapping]
    records: list[CanonicalMemberRecord]
    observations: list[dict[str, Any]]
    validation: dict[str, Any]

    def report(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id, "source_name": self.source_name,
            "source_format": self.source_format, "schema": self.schema,
            "mappings": [mapping.to_dict() for mapping in self.mappings],
            "record_count": len(self.records), "observations": self.observations,
            "validation": self.validation,
        }


def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_TABULAR_FORMATS | SUPPORTED_DOCUMENT_FORMATS:
        raise ValueError(f"Unsupported ingestion format: {suffix or 'no extension'}")
    return suffix


def extract_source(path: str | Path) -> tuple[pd.DataFrame | None, list[dict[str, Any]]]:
    source = Path(path)
    suffix = detect_format(source)
    if suffix == ".csv": return pd.read_csv(source), []
    if suffix in {".xlsx", ".xls"}: return pd.read_excel(source), []
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        return pd.json_normalize(payload), []
    if suffix in {".parquet", ".pq"}: return pd.read_parquet(source), []
    if suffix == ".txt":
        return None, [{"source_document": source.name, "section": "text", "text": source.read_text(encoding="utf-8"), "extracted_field": "document_text"}]
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX extraction requires python-docx.") from exc
        document = Document(source)
        return None, [{"source_document": source.name, "section": f"paragraph:{index + 1}", "text": paragraph.text, "extracted_field": "document_text"} for index, paragraph in enumerate(document.paragraphs) if paragraph.text.strip()]
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires pypdf.") from exc
    reader = PdfReader(source)
    return None, [{"source_document": source.name, "page": index + 1, "text": page.extract_text() or "", "extracted_field": "document_text"} for index, page in enumerate(reader.pages)]


def ingest_file(path: str | Path, schema: CanonicalSchema, *, source_name: str | None = None, aliases: Mapping[str, str] | None = None) -> IngestionResult:
    source = Path(path)
    dataframe, observations = extract_source(source)
    mapper = SemanticMapper(schema, aliases)
    dataset_id = str(uuid.uuid4())
    if dataframe is None:
        for observation in observations:
            observation.update({"dataset_id": dataset_id, "mapping_status": "UNMAPPED", "mapping_confidence": 0.0})
        return IngestionResult(dataset_id, source_name or source.name, detect_format(source), [], [], [], observations, {"passed": True, "warnings": ["Document observations are not training data."]})
    mappings = mapper.map_columns([str(column) for column in dataframe.columns])
    mapped = {entry.source_field: entry.canonical_feature for entry in mappings if entry.canonical_feature}
    member_sources = [source_field for source_field, target in mapped.items() if target == schema.member_id_column]
    records: list[CanonicalMemberRecord] = []
    errors: list[str] = []
    if len(member_sources) != 1:
        errors.append("Exactly one mapped member identifier is required.")
    else:
        member_source = member_sources[0]
        for index, row in dataframe.iterrows():
            raw_id = row[member_source]
            if _is_missing(raw_id) or not str(raw_id).strip():
                errors.append(f"row:{index}: missing member identifier")
                continue
            features = {target: row[source_field] for source_field, target in mapped.items() if target != schema.member_id_column and not _is_missing(row[source_field])}
            records.append(CanonicalMemberRecord(member_id=str(raw_id).strip(), source=source_name or source.name, source_record_id=str(index), provenance={"dataset_id": dataset_id, "source_file": source.name, "row_index": int(index), "canonical_features": features, "mappings": [entry.to_dict() for entry in mappings]}))
    validation = validate_canonical_members(records, schema).to_dict()
    validation["errors"] = [*errors, *validation["errors"]]
    validation["passed"] = not validation["errors"]
    validation["unmapped_fields"] = [entry.source_field for entry in mappings if entry.status == "UNMAPPED"]
    validation["review_required_fields"] = [entry.source_field for entry in mappings if entry.status == "REVIEW_REQUIRED"]
    return IngestionResult(dataset_id, source_name or source.name, detect_format(source), [str(column) for column in dataframe.columns], mappings, records, observations, validation)
