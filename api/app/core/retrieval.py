"""Main retrieval entry point.

This module now uses the Advanced Hybrid Retrieval Engine by default.
The original TF-IDF implementation is preserved as LegacyRetrievalEngine for reference.
"""

from __future__ import annotations

from .hybrid_retrieval import (
    HybridConfig,
    HybridRetrievalEngine,
    RetrievalResult,
)

# Alias for backward compatibility
RetrievalEngine = HybridRetrievalEngine

__all__ = [
    "RetrievalResult",
    "RetrievalEngine",
    "HybridRetrievalEngine",
    "HybridConfig",
]
