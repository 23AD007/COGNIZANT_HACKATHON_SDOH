"""
HealthLens canonical ingestion layer.

This package provides:
- canonical member representation
- source adapters
- schema validation
- canonical member dataset construction
"""

from .canonical_schema import (
    CanonicalMemberRecord,
    CanonicalSchema,
    load_existing_member_schema,
)

from .source_adapter import (
    SourceAdapter,
    DataFrameSourceAdapter,
)

from .validation import (
    ValidationResult,
    validate_canonical_members,
)
from .semantic_mapping import FieldMapping, SemanticMapper
from .external_ingestion import IngestionResult, detect_format, ingest_file

__all__ = [
    "CanonicalMemberRecord",
    "CanonicalSchema",
    "load_existing_member_schema",
    "SourceAdapter",
    "DataFrameSourceAdapter",
    "ValidationResult",
    "validate_canonical_members",
    "FieldMapping",
    "SemanticMapper",
    "IngestionResult",
    "detect_format",
    "ingest_file",
]
