"""Read-only Knowledge Base loading facade.

``registry.build_registry`` remains the authoritative loader.  This small
facade exists for workflow callers that need one validated KB object.
"""
from __future__ import annotations

from .registry import KnowledgeBaseRegistry, build_registry


def load_knowledge_base() -> KnowledgeBaseRegistry:
    """Return the validated, existing Knowledge Base registry."""
    return build_registry()


def load_registry() -> KnowledgeBaseRegistry:
    """Compatibility name for :func:`load_knowledge_base`."""
    return load_knowledge_base()


__all__ = ["load_knowledge_base", "load_registry", "KnowledgeBaseRegistry"]
