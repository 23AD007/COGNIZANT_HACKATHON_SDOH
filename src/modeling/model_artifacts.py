"""Compatibility access to the member-risk training artifact format."""
from __future__ import annotations

import pickle
from pathlib import Path

from .train_member_risk_model import MODEL_FILE, ModelArtifact, load_artifacts


class _MemberRiskArtifactUnpickler(pickle.Unpickler):
    """Read artifacts saved by either module or script execution."""

    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "ModelArtifact":
            return ModelArtifact
        return super().find_class(module, name)


def load_member_risk_artifact(path: str | Path | None = None) -> ModelArtifact:
    """Load the exact artifact written by ``train_member_risk_model``."""
    artifact_path = Path(path) if path is not None else MODEL_FILE
    try:
        return load_artifacts(artifact_path)
    except AttributeError as exc:
        # Older committed artifacts were written via ``python file.py`` and
        # therefore reference ``__main__.ModelArtifact`` in their pickle.
        if "ModelArtifact" not in str(exc):
            raise
        with artifact_path.open("rb") as handle:
            artifact = _MemberRiskArtifactUnpickler(handle).load()
        if not isinstance(artifact, ModelArtifact):
            raise TypeError("Invalid member-risk model artifact.")
        return artifact


load_model_artifacts = load_member_risk_artifact


__all__ = ["ModelArtifact", "load_member_risk_artifact", "load_model_artifacts"]
