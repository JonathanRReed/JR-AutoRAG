"""Gatherer executes retrieval plan steps and returns evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass

from .retrieval import RetrievalEngine, RetrievalResult


@dataclass
class EvidenceChunk:
    id: str
    title: str
    snippet: str
    score: float


@dataclass
class EvidenceBundle:
    chunks: list[EvidenceChunk]
    coverage: float
    token_estimate: int
    cache_info: dict[str, str]


class Gatherer:
    def __init__(self, retrieval: RetrievalEngine) -> None:
        self._retrieval = retrieval

    def gather(
        self,
        query: str,
        top_k: int,
        document_ids: list[str] | None = None,
        routing_params: dict | None = None,
    ) -> EvidenceBundle:
        results: list[RetrievalResult] = self._retrieval.query(
            query,
            top_k=top_k,
            document_ids=document_ids,
            routing_params=routing_params,
        )
        cache_info = {}
        if hasattr(self._retrieval, "get_last_cache_info"):
            cache_info = self._retrieval.get_last_cache_info()
        chunks = []
        for result in results:
            chunk_identifier = getattr(result, "chunk_id", None)
            if not chunk_identifier:
                start_marker = getattr(result, "start_char", 0)
                chunk_identifier = f"{result.document.id}-{start_marker}"
            snippet = getattr(result, "chunk_text", None) or result.document.text
            chunks.append(
                EvidenceChunk(
                    id=chunk_identifier,
                    title=result.document.title,
                    snippet=snippet,
                    score=result.score,
                )
            )
        coverage = min(1.0, len(chunks) / top_k) if top_k else 0.0
        token_estimate = sum(len(chunk.snippet.split()) for chunk in chunks)
        return EvidenceBundle(chunks=chunks, coverage=coverage, token_estimate=token_estimate, cache_info=cache_info)
