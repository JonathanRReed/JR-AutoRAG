"""Hybrid retrieval engine with optional binary-quantized search (Milvus HAMMING).

This wraps the existing HybridRetrievalEngine and adds a binary retrieval mode
that uses the in-process Milvus-compatible store for memory-efficient search.
"""

from __future__ import annotations

import asyncio
import contextvars
import time
from collections.abc import Callable
from typing import Any

from .binary_quantization import (
    BQ_VERSION,
    batch_float32_to_binary,
    float32_to_binary,
    validate_dimension,
)
from .binary_vector_store import MilvusChunk, MilvusSearchResult, MilvusVectorStore
from .bq_retrieval import BQRetrievalConfig, RetrievalDebug, RetrievalModeV2, RetrievalTimings
from .chunking import Chunk
from .documents import Document
from .hybrid_retrieval import HybridConfig, HybridRetrievalEngine, RetrievalResult

_last_bq_debug: contextvars.ContextVar[RetrievalDebug | None] = contextvars.ContextVar(
    "last_bq_debug", default=None
)


class BQHybridRetrievalEngine(HybridRetrievalEngine):
    """Hybrid retrieval engine with binary quantization mode.

    When binary mode is enabled, searches run against a binary vector store
    (HAMMING distance). Float32 hybrid retrieval is still available for
    fallback or when binary mode is disabled.
    """

    def __init__(
        self,
        documents,
        config: HybridConfig | None = None,
        bq_config: BQRetrievalConfig | None = None,
        bq_enabled: bool = False,
        persist_path: str | None = "data/indexes",
    ) -> None:
        super().__init__(documents, config=config, persist_path=persist_path)
        self._bq_config = bq_config or BQRetrievalConfig(default_mode=RetrievalModeV2.FLOAT32)
        self._bq_enabled = bool(bq_enabled) or self._bq_config.default_mode == RetrievalModeV2.BINARY
        self._bq_store: MilvusVectorStore | None = None
        self._bq_initialized = False
        self._bq_ready = False
        self._bq_embedding_dim = int(self._bq_config.embedding_dim or 0)
        self._last_bq_error: str | None = None

    def set_bq_config(
        self,
        config: BQRetrievalConfig,
        enabled: bool | None = None,
        rebuild: bool = True,
    ) -> None:
        """Update binary retrieval configuration and optionally rebuild the index."""
        previous = self._bq_config
        self._bq_config = config
        self._bq_enabled = bool(enabled) if enabled is not None else self._bq_enabled
        if self._bq_config.default_mode == RetrievalModeV2.BINARY:
            self._bq_enabled = True
        if not self._bq_enabled:
            self._bq_ready = False

        index_fields = ("embedding_dim", "embedding_model", "bq_config", "milvus_config")
        if any(getattr(previous, f) != getattr(self._bq_config, f) for f in index_fields):
            self._bq_store = None
            self._bq_initialized = False
            self._bq_ready = False

        if rebuild and self._bq_enabled and self._chunks and self._embeddings is not None:
            self._build_bq_index()

    def get_last_bq_debug(self) -> dict[str, Any]:
        """Return the last binary retrieval debug payload (if any)."""
        debug = _last_bq_debug.get()
        return debug.to_dict() if debug else {}

    def get_retrieval_mode_flags(self) -> int:
        mode = super().get_retrieval_mode_flags()
        try:
            from .cache import RetrievalMode
        except Exception:
            return mode
        if self._bq_enabled and self._bq_config.default_mode == RetrievalModeV2.BINARY:
            mode |= int(RetrievalMode.BINARY)
        return int(mode)

    def _ensure_bq_store(self, embedding_dim: int) -> MilvusVectorStore:
        """Ensure the binary store exists and matches the embedding dimension."""
        if self._bq_store is None or self._bq_embedding_dim != embedding_dim:
            self._bq_embedding_dim = embedding_dim
            self._bq_store = MilvusVectorStore(
                config=self._bq_config.milvus_config,
                embedding_dim=embedding_dim,
                bq_config=self._bq_config.bq_config,
            )
            self._bq_store.set_embedding_version(self._bq_config.embedding_model)
            self._bq_initialized = False
            self._bq_ready = False

        if not self._bq_initialized:
            self._bq_store.connect()
            self._bq_store.create_collection(drop_existing=True)
            self._bq_initialized = True

        return self._bq_store

    def _build_bq_index(self) -> None:
        """Build or rebuild the binary index from current chunks + embeddings."""
        self._last_bq_error = None
        if not self._bq_enabled:
            self._bq_ready = False
            return
        if not self._chunks or self._embeddings is None:
            self._bq_ready = False
            return

        embedding_dim = int(self._embeddings.shape[1])
        if not validate_dimension(embedding_dim):
            self._last_bq_error = (
                f"Embedding dim {embedding_dim} not divisible by 8; "
                "binary quantization disabled."
            )
            self._bq_ready = False
            return

        store = self._ensure_bq_store(embedding_dim)
        store.clear()

        docs = {doc.id: doc for doc in self._docs.list()}
        try:
            bq_vectors = batch_float32_to_binary(self._embeddings, self._bq_config.bq_config)
        except Exception as exc:
            self._last_bq_error = f"Binary quantization failed: {exc}"
            self._bq_ready = False
            return

        chunks: list[MilvusChunk] = []
        for idx, (doc_id, chunk) in enumerate(self._chunks):
            doc = docs.get(doc_id)
            if not doc:
                continue
            if idx >= len(bq_vectors):
                continue
            chunks.append(
                self._build_milvus_chunk(
                    doc=doc,
                    chunk=chunk,
                    bq_vector=bq_vectors[idx],
                )
            )

        if not chunks:
            self._bq_ready = False
            return

        store.bulk_insert(chunks)
        store.build_index()
        self._bq_ready = True

    def _reindex_bq_docs(self, doc_ids: set[str]) -> None:
        """Update binary index for a subset of documents."""
        if not self._bq_enabled or not doc_ids:
            return
        if self._embeddings is None or not self._chunks:
            self._bq_ready = False
            return

        embedding_dim = int(self._embeddings.shape[1])
        if not validate_dimension(embedding_dim):
            self._bq_ready = False
            return

        store = self._ensure_bq_store(embedding_dim)
        for doc_id in doc_ids:
            store.delete_by_doc_id(doc_id)

        docs = {doc.id: doc for doc in self._docs.list()}
        indices = [i for i, (doc_id, _) in enumerate(self._chunks) if doc_id in doc_ids]
        if not indices:
            self._bq_ready = True
            return

        try:
            bq_vectors = batch_float32_to_binary(self._embeddings[indices], self._bq_config.bq_config)
        except Exception:
            self._bq_ready = False
            return

        chunks: list[MilvusChunk] = []
        for local_idx, global_idx in enumerate(indices):
            doc_id, chunk = self._chunks[global_idx]
            doc = docs.get(doc_id)
            if not doc:
                continue
            chunks.append(
                self._build_milvus_chunk(
                    doc=doc,
                    chunk=chunk,
                    bq_vector=bq_vectors[local_idx],
                )
            )

        if chunks:
            store.bulk_insert(chunks)
            store.build_index()
            self._bq_ready = True

    def _build_milvus_chunk(
        self,
        doc: Document,
        chunk: Chunk,
        bq_vector: bytes,
    ) -> MilvusChunk:
        source = (
            doc.metadata.get("source")
            or doc.metadata.get("filename")
            or doc.title
        )
        metadata: dict[str, Any] = {
            "chunk_index": chunk.index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "doc_title": doc.title,
            "doc_metadata": dict(doc.metadata or {}),
        }
        if chunk.metadata:
            metadata.update(chunk.metadata)
        return MilvusChunk(
            doc_id=doc.id,
            chunk_id=f"{doc.id}-{chunk.index}",
            source=str(source),
            text=chunk.text,
            metadata=metadata,
            bq_vector=bq_vector,
        )

    def build(
        self,
        on_progress: Callable[[str, int, int, str | None], None] | None = None,
    ) -> None:
        super().build(on_progress=on_progress)
        if self._bq_enabled:
            self._build_bq_index()

    def clear(self) -> bool:
        cleared = super().clear()
        if self._bq_store:
            self._bq_store.clear()
        self._bq_initialized = False
        self._bq_ready = False
        return cleared

    def load_index(
        self,
        index_name: str = "default",
        on_progress: Callable[[str, int, int, str | None], None] | None = None,
    ) -> bool:
        loaded = super().load_index(index_name=index_name, on_progress=on_progress)
        if loaded and self._bq_enabled:
            self._build_bq_index()
        return loaded

    def index_documents(self, docs: list[Document]) -> None:
        super().index_documents(docs)
        if not self._bq_enabled:
            return
        doc_ids = {doc.id for doc in docs if doc.id}
        self._reindex_bq_docs(doc_ids)

    async def query(
        self,
        text: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        routing_params: dict[str, Any] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> list[RetrievalResult]:
        if not text.strip():
            _last_bq_debug.set(None)
            return []

        if not self._should_use_binary(routing_params):
            _last_bq_debug.set(None)
            return await super().query(
                text,
                top_k=top_k,
                document_ids=document_ids,
                routing_params=routing_params,
                on_progress=on_progress,
            )

        if not self._chunks:
            self.build()

        if not self._bq_ready:
            self._build_bq_index()

        if not self._bq_ready:
            _last_bq_debug.set(RetrievalDebug(
                mode="binary->float32",
                top_k=top_k,
                candidates_searched=0,
                results_returned=0,
                timings=RetrievalTimings(),
                fallback_triggered=True,
                fallback_reason=self._last_bq_error or "Binary index unavailable",
                embedding_version=self._bq_config.embedding_model,
                quantization_version=self._bq_config.bq_config.version,
            ))
            return await super().query(
                text,
                top_k=top_k,
                document_ids=document_ids,
                routing_params=routing_params,
                on_progress=on_progress,
            )

        loop = asyncio.get_event_loop()
        results, debug = await loop.run_in_executor(
            None,
            self._query_binary_sync,
            text,
            top_k,
            document_ids,
            routing_params,
        )
        _last_bq_debug.set(debug)
        return results

    def _should_use_binary(self, routing_params: dict[str, Any] | None) -> bool:
        if not self._bq_enabled:
            return False
        if routing_params and "retrieval_mode" in routing_params:
            mode = RetrievalModeV2.from_string(str(routing_params["retrieval_mode"]))
            return mode == RetrievalModeV2.BINARY
        return self._bq_config.default_mode == RetrievalModeV2.BINARY

    def _embed_query_cached(self, query: str) -> list[float] | None:
        if self._embedder is None:
            self._last_cache = {"embedding_cache": "skipped"}
            return None
        cached = self._cache_manager.embeddings.get(query)
        if cached is not None:
            self._last_cache = {"embedding_cache": "hit"}
            return list(cached)
        embedding = self._embedder.encode([query], convert_to_numpy=True)[0]
        self._cache_manager.embeddings.set(query, embedding.tolist())
        self._last_cache = {"embedding_cache": "miss"}
        return embedding.tolist()

    def _query_binary_sync(
        self,
        query: str,
        top_k: int,
        document_ids: list[str] | None,
        routing_params: dict[str, Any] | None,
    ) -> tuple[list[RetrievalResult], RetrievalDebug]:
        total_start = time.perf_counter()
        timings = RetrievalTimings()

        embed_start = time.perf_counter()
        embedding = self._embed_query_cached(query)
        timings.t_embed_query_ms = (time.perf_counter() - embed_start) * 1000

        if embedding is None:
            debug = self._fallback_debug(
                top_k,
                timings,
                "Embedding model unavailable",
            )
            return self._fallback_float32(query, top_k, document_ids, routing_params), debug

        quant_start = time.perf_counter()
        try:
            query_bq = float32_to_binary(embedding, self._bq_config.bq_config)
        except Exception as exc:
            timings.t_quantize_query_ms = (time.perf_counter() - quant_start) * 1000
            debug = self._fallback_debug(
                top_k,
                timings,
                f"Quantization failed: {exc}",
            )
            return self._fallback_float32(query, top_k, document_ids, routing_params), debug

        timings.t_quantize_query_ms = (time.perf_counter() - quant_start) * 1000

        search_k = top_k
        if self._bq_config.two_stage_enabled:
            search_k = max(search_k, self._bq_config.stage1_candidates)

        search_start = time.perf_counter()
        try:
            results = self._search_binary(query_bq, search_k, document_ids)
        except Exception as exc:
            timings.t_milvus_search_ms = (time.perf_counter() - search_start) * 1000
            debug = self._fallback_debug(
                top_k,
                timings,
                f"Binary search failed: {exc}",
            )
            return self._fallback_float32(query, top_k, document_ids, routing_params), debug
        timings.t_milvus_search_ms = (time.perf_counter() - search_start) * 1000

        if self._bq_config.fallback_enabled:
            if len(results) < self._bq_config.fallback_min_results:
                debug = self._fallback_debug(
                    top_k,
                    timings,
                    f"Insufficient results: {len(results)}",
                )
                return self._fallback_float32(query, top_k, document_ids, routing_params), debug
            if results and results[0].distance > self._bq_config.fallback_distance_threshold:
                debug = self._fallback_debug(
                    top_k,
                    timings,
                    f"High distance: {results[0].distance}",
                )
                return self._fallback_float32(query, top_k, document_ids, routing_params), debug

        reranked_scores: dict[str, float] | None = None
        if self._bq_config.two_stage_enabled and self._reranker and len(results) > top_k:
            rerank_start = time.perf_counter()
            reranked_scores = self._rerank_binary(query, results)
            timings.t_rerank_ms = (time.perf_counter() - rerank_start) * 1000
            results = [r for r, _ in sorted(
                zip(results, [reranked_scores.get(r.chunk_id, r.score) for r in results], strict=False),
                key=lambda item: item[1],
                reverse=True,
            )][:top_k]
        else:
            results = results[:top_k]

        context_start = time.perf_counter()
        retrieval_results = self._build_binary_results(results, reranked_scores)
        timings.t_context_build_ms = (time.perf_counter() - context_start) * 1000
        timings.t_total_ms = (time.perf_counter() - total_start) * 1000

        debug = RetrievalDebug(
            mode="binary",
            top_k=top_k,
            candidates_searched=search_k,
            results_returned=len(retrieval_results),
            timings=timings,
            distances=[r.distance for r in results],
            chunk_ids=[r.chunk_id for r in results],
            embedding_version=self._bq_config.embedding_model,
            quantization_version=self._bq_config.bq_config.version or BQ_VERSION,
        )

        return retrieval_results, debug

    def _search_binary(
        self,
        query_bq: bytes,
        top_k: int,
        document_ids: list[str] | None,
    ) -> list[MilvusSearchResult]:
        store = self._ensure_bq_store(self._bq_embedding_dim)
        return store.search_binary(query_bq, top_k=top_k, document_ids=document_ids)

    def _dedupe_by_chunk(self, results: list[MilvusSearchResult]) -> list[MilvusSearchResult]:
        by_id: dict[str, MilvusSearchResult] = {}
        for result in results:
            existing = by_id.get(result.chunk_id)
            if existing is None or result.distance < existing.distance:
                by_id[result.chunk_id] = result
        return sorted(by_id.values(), key=lambda r: r.distance)

    def _rerank_binary(self, query: str, results: list[MilvusSearchResult]) -> dict[str, float]:
        pairs = [(query, r.text) for r in results]
        scores = self._reranker.predict(pairs)
        return {result.chunk_id: float(score) for result, score in zip(results, scores, strict=False)}

    def _build_binary_results(
        self,
        results: list[MilvusSearchResult],
        rerank_scores: dict[str, float] | None,
    ) -> list[RetrievalResult]:
        docs = {doc.id: doc for doc in self._docs.list()}
        retrieval_results: list[RetrievalResult] = []

        for result in results:
            doc = docs.get(result.doc_id)
            if not doc:
                continue

            chunk_index = int(result.metadata.get("chunk_index", -1))
            extra_context: list[str] = []
            if self._config.raptor and chunk_index >= 0 and result.doc_id in self._trees:
                from .hierarchy import HierarchicalRetriever
                hr = HierarchicalRetriever(self._trees[result.doc_id])
                extra_context = hr.get_context_chain(str(chunk_index))

            chunk_text = result.text
            if extra_context:
                context_str = "\n".join(extra_context)
                chunk_text = f"[Hierarchy Context]\n{context_str}\n\n[Chunk Content]\n{result.text}"

            score = rerank_scores.get(result.chunk_id, result.score) if rerank_scores else result.score
            chunk_doc = Document(
                id=result.chunk_id,
                title=doc.title,
                text=chunk_text,
                metadata=doc.metadata,
            )
            retrieval_results.append(
                RetrievalResult(
                    document=chunk_doc,
                    score=float(score),
                    chunk_text=result.text,
                    retrieval_method="binary",
                    chunk_id=result.chunk_id,
                    start_char=int(result.metadata.get("start_char", 0)),
                    end_char=int(result.metadata.get("end_char", 0)),
                )
            )

        return retrieval_results

    def _fallback_float32(
        self,
        query: str,
        top_k: int,
        document_ids: list[str] | None,
        routing_params: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                super().query(
                    query,
                    top_k=top_k,
                    document_ids=document_ids,
                    routing_params=routing_params,
                )
            )
        finally:
            loop.close()

    def _fallback_debug(
        self,
        top_k: int,
        timings: RetrievalTimings,
        reason: str,
    ) -> RetrievalDebug:
        timings.t_total_ms = timings.t_total_ms or 0.0
        return RetrievalDebug(
            mode="binary->float32",
            top_k=top_k,
            candidates_searched=0,
            results_returned=0,
            timings=timings,
            fallback_triggered=True,
            fallback_reason=reason,
            embedding_version=self._bq_config.embedding_model,
            quantization_version=self._bq_config.bq_config.version or BQ_VERSION,
        )
