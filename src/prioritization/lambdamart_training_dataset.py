"""Training-dataset helpers for the existing LambdaMART ranker.

Labels intentionally use the ranker's existing weak-label definition from
the rule-based priority score.  No new target is introduced here.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .lambdamart_ranker import (
    LambdaMARTTrainingExample,
    RankingCandidate,
    build_training_examples,
    normalize_features,
)


def build_lambdamart_training_dataset(
    grouped_candidates: Mapping[str, Sequence[RankingCandidate]],
) -> list[LambdaMARTTrainingExample]:
    """Create examples using the established LambdaMART candidate schema."""
    return build_training_examples(grouped_candidates)


def validate_training_dataset(examples: Iterable[LambdaMARTTrainingExample]) -> list[LambdaMARTTrainingExample]:
    """Validate grouping and canonicalize feature values before training."""
    validated = list(examples)
    groups: dict[str, int] = {}
    for example in validated:
        if not str(example.group_id).strip() or not str(example.intervention_id).strip():
            raise ValueError("Each LambdaMART training example needs group_id and intervention_id.")
        example.features = normalize_features(example.features)
        groups[example.group_id] = groups.get(example.group_id, 0) + 1
    undersized = [group_id for group_id, count in groups.items() if count < 2]
    if undersized:
        raise ValueError("LambdaMART requires at least two candidates per group: " + ", ".join(sorted(undersized)))
    return validated


def save_training_dataset(examples: Iterable[LambdaMARTTrainingExample], path: str | Path) -> Path:
    """Persist only the existing ranker training-example schema."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = validate_training_dataset(examples)
    with output.open("w", encoding="utf-8") as handle:
        json.dump({"examples": [asdict(example) for example in rows]}, handle, indent=2)
    return output


def load_training_dataset(path: str | Path) -> list[LambdaMARTTrainingExample]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("examples", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("LambdaMART training dataset must contain an examples list.")
    return validate_training_dataset(LambdaMARTTrainingExample(**row) for row in rows)


__all__ = [
    "build_lambdamart_training_dataset", "validate_training_dataset",
    "save_training_dataset", "load_training_dataset",
]
