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
        evidence = None
        for step in plan.steps:
            evidence = self._gatherer.gather(step.query, top_k=step.dense_k)
        evidence = evidence or self._gatherer.gather(query, top_k=3)
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
        trace = self._telemetry.record(
            prompt=query,
            answer=answer,
            metrics={
                "chunks": len(chunks),
                "coverage": evidence.coverage if evidence else 0.0,
                "tokens": evidence.token_estimate if evidence else 0,
            },
        )
        metrics = {
            "chunks": len(chunks),
            "coverage": evidence.coverage,
            "tokens": evidence.token_estimate,
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
