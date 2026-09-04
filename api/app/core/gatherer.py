"""Gatherer executes retrieval plan steps and returns evidence bundles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .prompt_guard import wrap_ingested_content
from .retrieval import RetrievalEngine, RetrievalResult


@dataclass
class EvidenceChunk:
    id: str
    title: str
    snippet: str
    score: float
    doc_id: str | None = None


@dataclass
class EvidenceBundle:
    chunks: list[EvidenceChunk]
    coverage: float
    token_estimate: int
    cache_info: dict[str, str]


class Gatherer:
    def __init__(self, retrieval: RetrievalEngine) -> None:
        self._retrieval = retrieval

    async def gather(
        self,
        query: str,
        top_k: int,
        document_ids: list[str] | None = None,
        routing_params: dict | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> EvidenceBundle:
        results: list[RetrievalResult] = await self._retrieval.query(
            query,
            top_k=top_k,
            document_ids=document_ids,
            routing_params=routing_params,
            on_progress=on_progress,
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
            wrapped_snippet = wrap_ingested_content(snippet, chunk_identifier)
            chunks.append(
                EvidenceChunk(
                    id=chunk_identifier,
                    title=result.document.title,
                    snippet=wrapped_snippet,
                    score=result.score,
                    doc_id=result.document.id,
                )
            )
        coverage = min(1.0, len(chunks) / top_k) if top_k else 0.0
        token_estimate = sum(len(chunk.snippet.split()) for chunk in chunks)
        return EvidenceBundle(
            chunks=chunks,
            coverage=coverage,
            token_estimate=token_estimate,
            cache_info=cache_info,
        )
