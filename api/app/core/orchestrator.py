"""Orchestrator ties Planner, RetrievalEngine, and Providers together."""

from __future__ import annotations

from typing import List

from .planner import Planner
from .retrieval import RetrievalEngine
from .providers import ProviderFactory, LLMProvider, ProviderError
from .telemetry import TelemetryStore
from .gatherer import Gatherer
from ..schemas.config import AppConfig


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        retrieval: RetrievalEngine,
        gatherer: Gatherer,
        provider_factory: ProviderFactory,
        telemetry: TelemetryStore,
    ) -> None:
        self._planner = planner
        self._retrieval = retrieval
        self._gatherer = gatherer
        self._providers = provider_factory
        self._telemetry = telemetry
        self._provider: LLMProvider | None = None

    def rebuild(self, config: AppConfig) -> None:
        self._planner.rebuild(config)
        if config.provider:
            self._provider = self._providers.build(config.provider)
        self._retrieval.build()

    async def answer(self, query: str) -> dict:
        plan = self._planner.plan(query)
        all_chunks = []
        for step in plan.steps:
            step_evidence = self._gatherer.gather(step.query, top_k=step.dense_k)
            all_chunks.extend(step_evidence.chunks)
        
        # Deduplicate chunks by ID
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk.id not in seen:
                seen.add(chunk.id)
                unique_chunks.append(chunk)
        
        chunks = unique_chunks
        if not chunks:
             evidence = self._gatherer.gather(query, top_k=3)
             chunks = evidence.chunks
        provider = self._provider
        if provider is None:
            summary = "\n\n".join(chunk.snippet for chunk in chunks)
            answer = f"(No provider configured.) Context summary:\n{summary}" if summary else "No documents ingested yet."
        else:
            context = "\n\n".join(chunk.snippet for chunk in chunks)
            messages = [
                {"role": "system", "content": "You are JR AutoRAG assistant."},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ]
            try:
                answer = await provider.chat(messages)
            except ProviderError as exc:
                answer = f"Provider error: {exc}"
        
        # Calculate metrics from chunks
        total_tokens = sum(len(chunk.snippet.split()) for chunk in chunks)
        # Rough coverage estimate: ratio of chunks found to dense_k of first step (simplified)
        coverage = 0.0
        if plan.steps:
             coverage = min(1.0, len(chunks) / plan.steps[0].dense_k)
        
        trace = self._telemetry.record(
            prompt=query,
            answer=answer,
            metrics={
                "chunks": len(chunks),
                "coverage": coverage,
                "tokens": total_tokens,
            },
        )
        metrics = {
            "chunks": len(chunks),
            "coverage": coverage,
            "tokens": total_tokens,
        }
        return {
            "answer": answer,
            "chunks": [
                {
                    "id": chunk.id,
                    "title": chunk.title,
                    "score": chunk.score,
                    "snippet": chunk.snippet,
                }
                for chunk in chunks
            ],
            "trace_id": trace.id,
            "metrics": metrics,
        }
