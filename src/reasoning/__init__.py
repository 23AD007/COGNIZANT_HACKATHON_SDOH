"""
HealthLens Reasoning Layer
===========================

Public API for the contextual reasoning engine.

Architecture
------------
Knowledge Graph
        |
        v
Knowledge Base Registry
        |
        v
ContextualReasoner
        |
        v
MemberContext
        |
        v
ReasoningResult
        |
        +--> SDOH factors
        +--> SDOH domains
        +--> Clinical context
        +--> Evidence
        +--> Intervention candidates
        +--> Reasoning trace

This module intentionally contains only public exports.
It must not execute the contextual reasoner self-test during
package import.

This prevents the following warning when executing:

    py -3.12 -m src.reasoning.contextual_reasoner

    RuntimeWarning:
    'src.reasoning.contextual_reasoner' found in sys.modules
    after import of package 'src.reasoning'
"""

from .contextual_reasoner import (
    MemberContext,
    ReasoningResult,
    ContextualReasoner,
    load_graph,
    serialize_reasoning_result,
)


__all__ = [
    "MemberContext",
    "ReasoningResult",
    "ContextualReasoner",
    "load_graph",
    "serialize_reasoning_result",
]