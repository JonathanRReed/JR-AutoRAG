"""Backward-compatible Milvus store imports.

The canonical implementation lives in ``binary_vector_store.py``.
This module preserves older import paths used by tests and external callers.
"""

from .binary_vector_store import (
    IndexStats,
    MilvusChunk,
    MilvusConfig,
    MilvusSearchResult,
    MilvusVectorStore,
    get_milvus_store,
    is_milvus_available,
)

__all__ = [
    "MilvusConfig",
    "MilvusChunk",
    "MilvusSearchResult",
    "MilvusVectorStore",
    "IndexStats",
    "is_milvus_available",
    "get_milvus_store",
]
