"""Training entry point for the existing LambdaMART implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .lambdamart_ranker import (
    DEFAULT_MODEL_PATH,
    LambdaMARTModelMetadata,
    LambdaMARTRanker,
    LambdaMARTTrainingExample,
)
from .lambdamart_training_dataset import load_training_dataset, validate_training_dataset


def train_lambdamart(
    examples: Sequence[LambdaMARTTrainingExample],
    model_path: str | Path | None = None,
    **training_options: object,
) -> LambdaMARTModelMetadata:
    """Train and persist through ``LambdaMARTRanker``'s native artifact API."""
    ranker = LambdaMARTRanker(model_path=model_path or DEFAULT_MODEL_PATH)
    return ranker.train(validate_training_dataset(examples), **training_options)


def train_from_dataset(
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    **training_options: object,
) -> LambdaMARTModelMetadata:
    return train_lambdamart(load_training_dataset(dataset_path), model_path, **training_options)


def load_lambdamart_ranker(model_path: str | Path | None = None) -> LambdaMARTRanker:
    """Load the standard ranker artifact, retaining its fallback behavior."""
    return LambdaMARTRanker(model_path=model_path or DEFAULT_MODEL_PATH)


__all__ = ["train_lambdamart", "train_from_dataset", "load_lambdamart_ranker"]
