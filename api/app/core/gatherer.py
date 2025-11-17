"""Gatherer executes retrieval plan steps and returns evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .retrieval import RetrievalEngine, RetrievalResult


@dataclass
class EvidenceChunk:
    id: str
    title: str
    snippet: str
    score: float


@dataclass
class EvidenceBundle:
    chunks: List[EvidenceChunk]
    coverage: float
    token_estimate: int


class Gatherer:
    def __init__(self, retrieval: RetrievalEngine) -> None:
        self._retrieval = retrieval

    def gather(self, query: str, top_k: int) -> EvidenceBundle:
        results: List[RetrievalResult] = self._retrieval.query(query, top_k=top_k)
        chunks = [
            EvidenceChunk(
                id=result.document.id,
                title=result.document.title,
                snippet=result.document.text[:500],
                score=result.score,
            )
            for result in results
        ]
        coverage = min(1.0, len(chunks) / top_k) if top_k else 0.0
        token_estimate = sum(len(chunk.snippet.split()) for chunk in chunks)
        return EvidenceBundle(chunks=chunks, coverage=coverage, token_estimate=token_estimate)
