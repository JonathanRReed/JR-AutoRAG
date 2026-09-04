"""Binary Quantization Retrieval Service for JR AutoRAG v2.

This module provides:
- Dual-mode retrieval (binary/float32) with runtime switching
- Two-stage retrieval path (binary search + optional reranking)
- Integration with existing HybridRetrievalEngine
- Per-stage timing and debug payloads
- Feature flags and fallback mechanisms
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .binary_quantization import (
    BQ_VERSION,
    BQConfig,
)
from .binary_vector_store import (
    MilvusChunk,
    MilvusConfig,
    MilvusSearchResult,
    MilvusVectorStore,
    is_milvus_available,
)

if TYPE_CHECKING:
    from .hybrid_retrieval import HybridRetrievalEngine


class RetrievalModeV2(str, Enum):
    """Retrieval mode for v2 binary quantization."""

    FLOAT32 = "float32"
    BINARY = "binary"

    @classmethod
    def from_string(cls, value: str) -> RetrievalModeV2:
        if value.lower() in ("binary", "bq", "hamming"):
            return cls.BINARY
        return cls.FLOAT32


@dataclass
class RetrievalTimings:
    """Per-stage timing measurements for observability."""

    t_embed_query_ms: float = 0.0
    t_quantize_query_ms: float = 0.0
    t_milvus_search_ms: float = 0.0
    t_context_build_ms: float = 0.0
    t_rerank_ms: float = 0.0
    t_total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "t_embed_query_ms": self.t_embed_query_ms,
            "t_quantize_query_ms": self.t_quantize_query_ms,
            "t_milvus_search_ms": self.t_milvus_search_ms,
            "t_context_build_ms": self.t_context_build_ms,
            "t_rerank_ms": self.t_rerank_ms,
            "t_total_ms": self.t_total_ms,
        }


@dataclass
class RetrievalDebug:
    """Debug payload for retrieval operations."""

    mode: str
    top_k: int
    candidates_searched: int
    results_returned: int
    timings: RetrievalTimings
    distances: list[float] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    fallback_triggered: bool = False
    fallback_reason: str = ""
    embedding_version: str = ""
    quantization_version: str = BQ_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "candidates_searched": self.candidates_searched,
            "results_returned": self.results_returned,
            "timings": self.timings.to_dict(),
            "distances": self.distances,
            "chunk_ids": self.chunk_ids,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "embedding_version": self.embedding_version,
            "quantization_version": self.quantization_version,
        }


@dataclass
class RetrievedChunk:
    """A retrieved chunk with metadata for generation layer."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class BQRetrievalConfig:
    """Configuration for BQ retrieval service."""

    # Mode selection
    default_mode: RetrievalModeV2 = RetrievalModeV2.BINARY

    # Search parameters
    top_k: int = 5

    # Two-stage retrieval (binary search + rerank)
    two_stage_enabled: bool = False
    stage1_candidates: int = 50  # Binary search returns top N

    # Fallback settings
    fallback_enabled: bool = True
    fallback_distance_threshold: float = 500.0  # Hamming distance threshold
    fallback_min_results: int = 1

    # Milvus settings
    milvus_config: MilvusConfig = field(default_factory=MilvusConfig)

    # Binary quantization settings
    bq_config: BQConfig = field(default_factory=BQConfig)

    # Embedding settings
    embedding_dim: int = 768
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_mode": self.default_mode.value,
            "top_k": self.top_k,
            "two_stage_enabled": self.two_stage_enabled,
            "stage1_candidates": self.stage1_candidates,
            "fallback_enabled": self.fallback_enabled,
            "fallback_distance_threshold": self.fallback_distance_threshold,
            "fallback_min_results": self.fallback_min_results,
            "milvus_config": self.milvus_config.to_dict(),
            "bq_config": self.bq_config.to_dict(),
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model,
        }


class BQRetrievalService:
    """Binary Quantization Retrieval Service.

    Provides dual-mode retrieval with binary (Milvus HAMMING) and float32 (existing engine).
    Implements the JR AutoRAG v2 design for memory-efficient retrieval.
    """

    def __init__(
        self,
        config: BQRetrievalConfig | None = None,
        float32_engine: HybridRetrievalEngine | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        """Initialize BQ retrieval service.

        Args:
            config: Service configuration
            float32_engine: Existing HybridRetrievalEngine for fallback/float32 mode
            embed_fn: Function to embed query text (uses float32_engine if not provided)
        """
        self._config = config or BQRetrievalConfig()
        self._float32_engine = float32_engine
        self._embed_fn = embed_fn

        # Milvus store (lazy initialized)
        self._milvus_store: MilvusVectorStore | None = None
        self._milvus_initialized = False

        # Reranker (from float32 engine if available)
        self._reranker = None
        if float32_engine and hasattr(float32_engine, "_reranker"):
            self._reranker = float32_engine._reranker

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> list[str]:
        """Split plain text into deterministic chunks for BQ directory indexing."""
        normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        if not normalized:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + max_chars)
            if end < len(normalized):
                boundary = max(
                    normalized.rfind("\n\n", start, end),
                    normalized.rfind(". ", start, end),
                    normalized.rfind("\n", start, end),
                )
                if boundary > start + max_chars // 2:
                    end = boundary + 1
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def _load_chunks_from_docs_path(
        self, docs_path: str
    ) -> tuple[list[MilvusChunk], dict[str, Any]]:
        """Load text and Markdown files from a directory into embedded Milvus chunks."""
        root = Path(docs_path).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"docs_path does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"docs_path is not a directory: {root}")

        supported_suffixes = {".txt", ".md", ".markdown"}
        files = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in supported_suffixes
        ]
        chunks: list[MilvusChunk] = []
        skipped_empty = 0
        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            text_chunks = self._chunk_text(text)
            if not text_chunks:
                skipped_empty += 1
                continue
            rel_source = str(file_path.relative_to(root))
            doc_id = sha256(rel_source.encode("utf-8")).hexdigest()[:16]
            for index, chunk_text in enumerate(text_chunks):
                embedding, _ = self._embed_query(chunk_text)
                chunks.append(
                    MilvusChunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}-{index}",
                        source=rel_source,
                        text=chunk_text,
                        metadata={
                            "source_path": rel_source,
                            "chunk_index": index,
                            "index_source": "docs_path",
                        },
                        embedding=embedding,
                    )
                )

        return chunks, {
            "documents_scanned": len(files),
            "documents_skipped_empty": skipped_empty,
            "supported_extensions": sorted(supported_suffixes),
        }

    def _ensure_milvus(self) -> MilvusVectorStore:
        """Ensure Milvus store is initialized."""
        if self._milvus_store is None:
            if not is_milvus_available():
                raise RuntimeError(
                    "pymilvus not installed. Install with: cd api && uv pip install pymilvus"
                )

            self._milvus_store = MilvusVectorStore(
                config=self._config.milvus_config,
                embedding_dim=self._config.embedding_dim,
                bq_config=self._config.bq_config,
            )
            self._milvus_store.set_embedding_version(self._config.embedding_model)

        if not self._milvus_initialized:
            self._milvus_store.connect()
            self._milvus_store.create_collection()
            self._milvus_initialized = True

        return self._milvus_store

    def _embed_query(self, query: str) -> tuple[list[float], float]:
        """Embed query text and return timing.

        Returns:
            Tuple of (embedding, time_ms)
        """
        start = time.perf_counter()

        if self._embed_fn:
            embedding = self._embed_fn(query)
        elif self._float32_engine and hasattr(self._float32_engine, "_embedder"):
            embedder = self._float32_engine._embedder
            if embedder:
                embedding = embedder.encode(query).tolist()
            else:
                raise RuntimeError("No embedding model available")
        else:
            raise RuntimeError("No embedding function or engine available")

        elapsed_ms = (time.perf_counter() - start) * 1000
        return embedding, elapsed_ms

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        mode: RetrievalModeV2 | str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        """Retrieve relevant chunks for a query.

        Args:
            query: User query text
            k: Number of results (default from config)
            mode: Retrieval mode override (binary/float32)
            filters: Optional metadata filters

        Returns:
            Tuple of (retrieved_chunks, debug_info)
        """
        total_start = time.perf_counter()
        timings = RetrievalTimings()

        # Resolve parameters
        k = k or self._config.top_k
        if mode is None:
            mode = self._config.default_mode
        elif isinstance(mode, str):
            mode = RetrievalModeV2.from_string(mode)

        # Embed query
        embedding, embed_time = self._embed_query(query)
        timings.t_embed_query_ms = embed_time

        # Route to appropriate retrieval mode
        if mode == RetrievalModeV2.BINARY:
            chunks, debug = self._retrieve_binary(query, embedding, k, filters, timings)
        else:
            chunks, debug = self._retrieve_float32(
                query, embedding, k, filters, timings
            )

        timings.t_total_ms = (time.perf_counter() - total_start) * 1000
        debug.timings = timings

        return chunks, debug

    def _retrieve_binary(
        self,
        query: str,
        embedding: list[float],
        k: int,
        filters: dict[str, Any] | None,
        timings: RetrievalTimings,
    ) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        """Retrieve using binary quantization and Milvus HAMMING search."""

        # Quantize query
        quant_start = time.perf_counter()
        # Quantization happens inside search, but we measure it separately
        timings.t_quantize_query_ms = (time.perf_counter() - quant_start) * 1000

        # Determine search count (two-stage or direct)
        search_k = k
        if self._config.two_stage_enabled:
            search_k = self._config.stage1_candidates

        # Search Milvus
        search_start = time.perf_counter()
        try:
            store = self._ensure_milvus()

            # Build filter expression
            filter_expr = None
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_parts.append(f'{key} == "{value}"')
                    else:
                        filter_parts.append(f"{key} == {value}")
                if filter_parts:
                    filter_expr = " and ".join(filter_parts)

            results = store.search(embedding, top_k=search_k, filter_expr=filter_expr)
            timings.t_milvus_search_ms = (time.perf_counter() - search_start) * 1000

        except Exception as e:
            # Fallback to float32 if Milvus fails
            if self._config.fallback_enabled and self._float32_engine:
                return self._fallback_to_float32(
                    query, embedding, k, filters, timings, reason=f"Milvus error: {e}"
                )
            raise

        # Check if fallback needed (empty results or high distances)
        if self._config.fallback_enabled:
            if len(results) < self._config.fallback_min_results:
                return self._fallback_to_float32(
                    query,
                    embedding,
                    k,
                    filters,
                    timings,
                    reason=f"Insufficient results: {len(results)}",
                )

            if (
                results
                and results[0].distance > self._config.fallback_distance_threshold
            ):
                return self._fallback_to_float32(
                    query,
                    embedding,
                    k,
                    filters,
                    timings,
                    reason=f"High distance: {results[0].distance}",
                )

        # Two-stage reranking if enabled
        if self._config.two_stage_enabled and self._reranker and len(results) > k:
            rerank_start = time.perf_counter()
            results = self._rerank_results(query, results, k)
            timings.t_rerank_ms = (time.perf_counter() - rerank_start) * 1000
        else:
            results = results[:k]

        # Build context
        context_start = time.perf_counter()
        chunks = self._build_chunks(results)
        timings.t_context_build_ms = (time.perf_counter() - context_start) * 1000

        # Build debug info
        debug = RetrievalDebug(
            mode="binary",
            top_k=k,
            candidates_searched=search_k,
            results_returned=len(chunks),
            timings=timings,
            distances=[r.distance for r in results],
            chunk_ids=[r.chunk_id for r in results],
            embedding_version=self._config.embedding_model,
            quantization_version=self._config.bq_config.version,
        )

        return chunks, debug

    def _retrieve_float32(
        self,
        query: str,
        embedding: list[float],
        k: int,
        filters: dict[str, Any] | None,
        timings: RetrievalTimings,
    ) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        """Retrieve using existing float32 HybridRetrievalEngine."""

        if not self._float32_engine:
            raise RuntimeError("Float32 engine not available")

        search_start = time.perf_counter()

        # Use existing engine's retrieve method
        results = self._float32_engine.retrieve(query, k=k)

        timings.t_milvus_search_ms = (time.perf_counter() - search_start) * 1000

        # Convert to RetrievedChunk format
        context_start = time.perf_counter()
        chunks = []
        for result in results:
            chunks.append(
                RetrievedChunk(
                    chunk_id=result.chunk_id,
                    doc_id=result.document.id if result.document else "",
                    text=result.chunk_text,
                    score=result.score,
                    source=result.document.title if result.document else "",
                    metadata=result.document.metadata if result.document else {},
                    start_char=result.start_char,
                    end_char=result.end_char,
                )
            )
        timings.t_context_build_ms = (time.perf_counter() - context_start) * 1000

        debug = RetrievalDebug(
            mode="float32",
            top_k=k,
            candidates_searched=k,
            results_returned=len(chunks),
            timings=timings,
            distances=[],
            chunk_ids=[c.chunk_id for c in chunks],
            embedding_version=self._config.embedding_model,
        )

        return chunks, debug

    def _fallback_to_float32(
        self,
        query: str,
        embedding: list[float],
        k: int,
        filters: dict[str, Any] | None,
        timings: RetrievalTimings,
        reason: str,
    ) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        """Fallback to float32 retrieval."""

        if not self._float32_engine:
            # Return empty results if no fallback available
            debug = RetrievalDebug(
                mode="binary",
                top_k=k,
                candidates_searched=0,
                results_returned=0,
                timings=timings,
                fallback_triggered=True,
                fallback_reason=f"{reason} (no fallback engine)",
            )
            return [], debug

        chunks, debug = self._retrieve_float32(query, embedding, k, filters, timings)
        debug.fallback_triggered = True
        debug.fallback_reason = reason
        debug.mode = "binary->float32"

        return chunks, debug

    def _rerank_results(
        self,
        query: str,
        results: list[MilvusSearchResult],
        k: int,
    ) -> list[MilvusSearchResult]:
        """Rerank results using cross-encoder."""

        if not self._reranker:
            return results[:k]

        # Prepare pairs for reranking
        pairs = [(query, r.text) for r in results]

        try:
            scores = self._reranker.predict(pairs)

            # Sort by reranker score (descending)
            scored_results = list(zip(results, scores, strict=False))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            return [r for r, _ in scored_results[:k]]
        except Exception as e:
            print(f"Reranking failed: {e}")
            return results[:k]

    def _build_chunks(self, results: list[MilvusSearchResult]) -> list[RetrievedChunk]:
        """Convert Milvus results to RetrievedChunk format."""
        chunks = []
        for result in results:
            chunks.append(
                RetrievedChunk(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    text=result.text,
                    score=result.score,
                    source=result.source,
                    metadata=result.metadata,
                )
            )
        return chunks

    def index_documents(
        self,
        docs_path: str | None = None,
        chunks: list[MilvusChunk] | None = None,
        mode: RetrievalModeV2 = RetrievalModeV2.BINARY,
    ) -> dict[str, Any]:
        """Index documents into the retrieval store.

        Args:
            docs_path: Path to a directory of .txt, .md, or .markdown documents
            chunks: Pre-processed chunks to index
            mode: Indexing mode

        Returns:
            Indexing statistics
        """
        start = time.perf_counter()

        if mode == RetrievalModeV2.BINARY:
            docs_stats: dict[str, Any] = {}
            if not chunks and docs_path:
                try:
                    chunks, docs_stats = self._load_chunks_from_docs_path(docs_path)
                except Exception as exc:
                    return {
                        "mode": "binary",
                        "chunks_indexed": 0,
                        "error": str(exc),
                    }

            if chunks:
                store = self._ensure_milvus()
                ids = store.bulk_insert(chunks)
                store.build_index()

                elapsed = (time.perf_counter() - start) * 1000
                return {
                    "mode": "binary",
                    "chunks_indexed": len(ids),
                    "elapsed_ms": elapsed,
                    "collection": self._config.milvus_config.collection_name,
                    **docs_stats,
                }

            return {
                "mode": "binary",
                "chunks_indexed": 0,
                "error": "No chunks or supported docs_path documents provided",
            }

        else:
            # Use existing float32 engine
            if self._float32_engine:
                self._float32_engine.build()
                elapsed = (time.perf_counter() - start) * 1000
                return {
                    "mode": "float32",
                    "elapsed_ms": elapsed,
                }
            return {"mode": "float32", "error": "No float32 engine available"}

    def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the current index."""
        stats = {}

        # Binary index stats
        if self._milvus_store and self._milvus_initialized:
            milvus_stats = self._milvus_store.get_stats()
            stats["binary"] = {
                "count": milvus_stats.count,
                "index_type": milvus_stats.index_type,
                "metric": milvus_stats.metric,
                "dim_bits": milvus_stats.dim_bits,
                "storage_estimate_bytes": milvus_stats.storage_estimate_bytes,
                "collection": milvus_stats.collection_name,
            }

        # Float32 index stats
        if self._float32_engine:
            stats["float32"] = {
                "chunks": len(self._float32_engine._chunks)
                if hasattr(self._float32_engine, "_chunks")
                else 0,
            }

        return stats

    def clear(self, mode: RetrievalModeV2 | None = None) -> None:
        """Clear index data.

        Args:
            mode: Specific mode to clear, or None for all
        """
        if (mode is None or mode == RetrievalModeV2.BINARY) and self._milvus_store:
            self._milvus_store.clear()
            self._milvus_initialized = False

        if (
            (mode is None or mode == RetrievalModeV2.FLOAT32)
            and self._float32_engine
            and hasattr(self._float32_engine, "clear")
        ):
            self._float32_engine.clear()


def get_bq_retrieval_service(
    config: BQRetrievalConfig | None = None,
    float32_engine: HybridRetrievalEngine | None = None,
) -> BQRetrievalService:
    """Factory function for BQ retrieval service."""
    return BQRetrievalService(config=config, float32_engine=float32_engine)


__all__ = [
    "RetrievalModeV2",
    "RetrievalTimings",
    "RetrievalDebug",
    "RetrievedChunk",
    "BQRetrievalConfig",
    "BQRetrievalService",
    "get_bq_retrieval_service",
]
