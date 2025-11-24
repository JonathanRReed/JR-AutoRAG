"""Orchestrator ties Planner, RetrievalEngine, and Providers together."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List

from .planner import Planner
from .retrieval import RetrievalEngine
from .providers import ProviderFactory, LLMProvider, ProviderError
from .telemetry import TelemetryStore, PipelineStep
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

    def _make_step(
        self, name: str, start: float, details: Dict[str, Any], status: str = "completed"
    ) -> PipelineStep:
        """Helper to create a PipelineStep with timing."""
        end = time.perf_counter()
        started_at = datetime.utcnow()
        return PipelineStep(
            name=name,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            duration_ms=round((end - start) * 1000, 2),
            details=details,
            status=status,
        )

    async def answer(self, query: str) -> dict:
        pipeline_start = datetime.utcnow()
        pipeline_steps: List[PipelineStep] = []

        # Step 1: Planning
        plan_start = time.perf_counter()
        plan = self._planner.plan(query)
        pipeline_steps.append(self._make_step(
            "planning",
            plan_start,
            {
                "num_steps": len(plan.steps),
                "target_tokens": plan.target_tokens,
                "coverage_target": plan.coverage_target,
                "queries": [s.query for s in plan.steps],
            },
        ))

        # Step 2: Retrieval
        retrieval_start = time.perf_counter()
        all_chunks = []
        retrieval_details: Dict[str, Any] = {"sub_queries": []}
        for step in plan.steps:
            sub_start = time.perf_counter()
            step_evidence = self._gatherer.gather(step.query, top_k=step.dense_k)
            all_chunks.extend(step_evidence.chunks)
            retrieval_details["sub_queries"].append({
                "query": step.query,
                "top_k": step.dense_k,
                "chunks_found": len(step_evidence.chunks),
                "duration_ms": round((time.perf_counter() - sub_start) * 1000, 2),
            })

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

        retrieval_details["total_chunks"] = len(chunks)
        retrieval_details["unique_sources"] = len({c.title for c in chunks})
        pipeline_steps.append(self._make_step("retrieval", retrieval_start, retrieval_details))

        # Step 3: Generation
        gen_start = time.perf_counter()
        provider = self._provider
        gen_details: Dict[str, Any] = {"provider": None, "model": None}

        if provider is None:
            summary = "\n\n".join(chunk.snippet for chunk in chunks)
            answer = f"(No provider configured.) Context summary:\n{summary}" if summary else "No documents ingested yet."
            gen_details["provider"] = "none"
            gen_details["fallback"] = True
        else:
            context = "\n\n".join(chunk.snippet for chunk in chunks)
            messages = [
                {"role": "system", "content": "You are JR AutoRAG assistant. Answer based on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
            gen_details["provider"] = getattr(provider, "base_url", "unknown")
            gen_details["model"] = getattr(provider, "default_model", "unknown")
            gen_details["context_tokens"] = sum(len(c.snippet.split()) for c in chunks)
            try:
                answer = await provider.chat(messages)
                gen_details["status"] = "success"
            except ProviderError as exc:
                answer = f"Provider error: {exc}"
                gen_details["status"] = "error"
                gen_details["error"] = str(exc)

        pipeline_steps.append(self._make_step("generation", gen_start, gen_details))

        # Calculate final metrics
        total_tokens = sum(len(chunk.snippet.split()) for chunk in chunks)
        coverage = 0.0
        if plan.steps:
            coverage = min(1.0, len(chunks) / plan.steps[0].dense_k)

        total_duration_ms = sum(s.duration_ms for s in pipeline_steps)

        trace = self._telemetry.record(
            prompt=query,
            answer=answer,
            metrics={
                "chunks": len(chunks),
                "coverage": coverage,
                "tokens": total_tokens,
                "duration_ms": total_duration_ms,
            },
            steps=pipeline_steps,
            started_at=pipeline_start,
        )

        # Build step summaries for response
        steps_out = [
            {
                "name": s.name,
                "duration_ms": s.duration_ms,
                "details": s.details,
                "status": s.status,
            }
            for s in pipeline_steps
        ]

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
            "metrics": {
                "chunks": len(chunks),
                "coverage": coverage,
                "tokens": total_tokens,
                "duration_ms": total_duration_ms,
            },
            "steps": steps_out,
        }
