"""Orchestrator ties Planner, RetrievalEngine, and Providers together."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from typing import Any
from collections.abc import Callable

from ..schemas.config import AppConfig
from .cache import get_cache_manager
from .gatherer import Gatherer
from .planner import Planner
from .providers import LLMProvider, ProviderError, ProviderFactory
from .retrieval import RetrievalEngine
from .reflection import SelfReflector
from .telemetry import PipelineStep, TelemetryStore
from .compression import ContextCompressor, CompressedContext


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
        self._compressor = ContextCompressor()
        self._reflector = SelfReflector()
        self._config: AppConfig | None = None

    def rebuild(self, config: AppConfig) -> None:
        self._config = config
        self._planner.rebuild(config)
        if config.provider:
            self._provider = self._providers.build(config.provider)
        if hasattr(self._planner, "set_provider"):
            self._planner.set_provider(self._provider)
        self._retrieval.build()
        # Update compressor with config settings
        self._compressor = ContextCompressor(
            max_tokens=config.retrieval.max_context_tokens,
        )

    def set_planner(self, planner: Planner) -> None:
        self._planner = planner

    def _make_step(
        self,
        name: str,
        start_perf: float,
        start_time: datetime,
        details: dict[str, Any],
        status: str = "completed",
    ) -> PipelineStep:
        """Helper to create a PipelineStep with timing."""
        end = time.perf_counter()
        return PipelineStep(
            name=name,
            started_at=start_time,
            completed_at=datetime.utcnow(),
            duration_ms=round((end - start_perf) * 1000, 2),
            details=details,
            status=status,
        )

    def _cache_config_hash(self, document_ids: list[str] | None) -> str:
        config = self._config
        provider = config.provider if config else None
        payload = {
            "document_ids": document_ids or [],
            "retrieval": config.retrieval.model_dump() if config else {},
            "provider": {
                "name": provider.name if provider else "",
                "base_url": str(provider.base_url) if provider else "",
                "planner_model": provider.planner_model if provider else "",
                "gatherer_model": provider.gatherer_model if provider else "",
                "generator_model": provider.generator_model if provider else "",
            },
        }
        encoded = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    async def answer(
        self,
        query: str,
        document_ids: list[str] | None = None,
        on_step: Callable[[PipelineStep], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> dict:
        pipeline_start = datetime.utcnow()
        pipeline_steps: list[PipelineStep] = []
        reflection_result = None

        def record_step(step: PipelineStep) -> None:
            pipeline_steps.append(step)
            if on_step:
                on_step(step)

        cache_manager = get_cache_manager()
        cache_hash = self._cache_config_hash(document_ids)
        cache_start_time = datetime.utcnow()
        cache_start = time.perf_counter()
        if on_stage:
            on_stage("cache")
        cached_result = cache_manager.queries.get(query, cache_hash)
        if cached_result:
            cache_step = self._make_step(
                "cache",
                cache_start,
                cache_start_time,
                {"query_cache": "hit", "cache_key": cache_hash},
            )
            record_step(cache_step)
            metrics = cached_result.get("metrics", {}) if isinstance(cached_result, dict) else {}
            metrics["cache_hit"] = True
            cached_steps = cached_result.get("steps", []) if isinstance(cached_result, dict) else []
            response = {
                **cached_result,
                "metrics": metrics,
                "steps": [
                    {
                        "name": cache_step.name,
                        "duration_ms": cache_step.duration_ms,
                        "details": cache_step.details,
                        "status": cache_step.status,
                        "started_at": cache_step.started_at.isoformat(),
                        "completed_at": cache_step.completed_at.isoformat(),
                    },
                    *cached_steps,
                ],
            }
            self._telemetry.record(
                prompt=query,
                answer=cached_result.get("answer", ""),
                metrics=metrics,
                steps=pipeline_steps,
                started_at=pipeline_start,
            )
            return response
        cache_step = self._make_step(
            "cache",
            cache_start,
            cache_start_time,
            {"query_cache": "miss", "cache_key": cache_hash},
        )
        record_step(cache_step)

        # Step 1: Planning (now with query analysis)
        if on_stage:
            on_stage("planning")
        plan_start_time = datetime.utcnow()
        plan_start = time.perf_counter()
        planner_mode = "heuristic"
        try:
            if hasattr(self._planner, "plan_async"):
                plan = await self._planner.plan_async(query)
                planner_mode = getattr(self._planner, "_last_planner_mode", "llm")
            else:
                plan = self._planner.plan(query)
        except Exception:
            plan = self._planner.plan(query)
            planner_mode = "heuristic"
        
        # Get query type if using SmartPlanner
        query_type = getattr(plan, 'query_type', 'factual')
        decomposed = getattr(plan, 'decomposed', False)
        
        record_step(self._make_step(
            "planning",
            plan_start,
            plan_start_time,
            {
                "num_steps": len(plan.steps),
                "target_tokens": plan.target_tokens,
                "coverage_target": plan.coverage_target,
                "queries": [s.query for s in plan.steps],
                "query_type": str(query_type),
                "decomposed": decomposed,
                "planner_mode": planner_mode,
                "expanded_terms": getattr(plan, "expanded_terms", []),
            },
        ))

        # Step 2: Gatherer (evidence collection)
        if on_stage:
            on_stage("gatherer")
        gatherer_start_time = datetime.utcnow()
        gatherer_start = time.perf_counter()
        all_chunks = []
        gatherer_details: dict[str, Any] = {"sub_queries": [], "literal_hits": 0}
        embedding_cache_hits = 0
        embedding_cache_misses = 0

        for step in plan.steps:
            sub_start = time.perf_counter()
            step_evidence = self._gatherer.gather(step.query, top_k=step.dense_k, document_ids=document_ids)
            all_chunks.extend(step_evidence.chunks)
            embedding_cache = step_evidence.cache_info.get("embedding_cache")
            literal_hits = step_evidence.cache_info.get("literal_hits", 0)
            if embedding_cache == "hit":
                embedding_cache_hits += 1
            elif embedding_cache == "miss":
                embedding_cache_misses += 1
            if isinstance(literal_hits, int):
                gatherer_details["literal_hits"] += literal_hits
            gatherer_details["sub_queries"].append({
                "query": step.query,
                "top_k": step.dense_k,
                "chunks_found": len(step_evidence.chunks),
                "duration_ms": round((time.perf_counter() - sub_start) * 1000, 2),
                "embedding_cache": embedding_cache,
                "literal_hits": literal_hits,
            })

        gatherer_details["embedding_cache_hits"] = embedding_cache_hits
        gatherer_details["embedding_cache_misses"] = embedding_cache_misses
        record_step(self._make_step("gatherer", gatherer_start, gatherer_start_time, gatherer_details))

        # Step 3: Retrieval (aggregation + dedupe)
        if on_stage:
            on_stage("retrieval")
        retrieval_start_time = datetime.utcnow()
        retrieval_start = time.perf_counter()
        retrieval_details: dict[str, Any] = {"sub_queries": gatherer_details["sub_queries"]}

        # Deduplicate chunks by ID, keeping highest score
        seen: dict[str, Any] = {}
        for chunk in all_chunks:
            if chunk.id not in seen or chunk.score > seen[chunk.id].score:
                seen[chunk.id] = chunk
        
        chunks = list(seen.values())
        # Sort by score descending
        chunks.sort(key=lambda c: c.score, reverse=True)
        
        if not chunks:
            evidence = self._gatherer.gather(query, top_k=3, document_ids=document_ids)
            chunks = evidence.chunks

        if document_ids:
            retrieval_details["document_filter"] = document_ids
        retrieval_details["total_chunks"] = len(chunks)
        retrieval_details["unique_sources"] = len({c.title for c in chunks})
        retrieval_details["embedding_cache_hits"] = embedding_cache_hits
        retrieval_details["embedding_cache_misses"] = embedding_cache_misses
        if hasattr(self._retrieval, "model_status"):
            retrieval_details.update(self._retrieval.model_status())
        record_step(self._make_step("retrieval", retrieval_start, retrieval_start_time, retrieval_details))

        # Step 3: Context Compression (new!)
        if on_stage:
            on_stage("compression")
        compression_start_time = datetime.utcnow()
        compression_start = time.perf_counter()
        use_compression = self._config and self._config.retrieval.compression
        
        if use_compression and chunks:
            compressed = self._compressor.compress(
                chunks, 
                query=query,
                strategy="extractive",
            )
            context = compressed.text
            citations = compressed.citations
            compression_details = {
                "enabled": True,
                "chunks_used": compressed.chunks_used,
                "chunks_total": compressed.chunks_total,
                "estimated_tokens": compressed.estimated_tokens,
            }
            compression_status = "completed"
        else:
            # Simple formatting with citations
            context, citations = self._compressor.format_with_citations(chunks)
            compression_details = {
                "enabled": False,
                "chunks_used": len(chunks),
                "chunks_total": len(chunks),
            }
            compression_status = "skipped"
        
        record_step(
            self._make_step(
                "compression",
                compression_start,
                compression_start_time,
                compression_details,
                status=compression_status,
            )
        )

        # Step 4: Generation (with citation prompt)
        if on_stage:
            on_stage("generation")
        gen_start_time = datetime.utcnow()
        gen_start = time.perf_counter()
        provider = self._provider
        gen_details: dict[str, Any] = {"provider": None, "model": None}

        if provider is None:
            answer = f"(No provider configured.) Context summary:\n{context}" if context else "No documents ingested yet."
            gen_details["provider"] = "none"
            gen_details["fallback"] = True
        else:
            # Enhanced prompt with citation instructions
            system_prompt = """You are JR AutoRAG assistant, a precise enterprise RAG generator.
You must answer using only the provided context and cite sources with bracketed numbers, e.g., [1], [2].
If the context does not contain the answer, say so clearly and suggest what is missing.
Be concise, structured, and avoid speculation."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
            gen_details["provider"] = getattr(provider, "base_url", "unknown")
            gen_details["model"] = getattr(provider, "default_model", "unknown")
            gen_details["context_tokens"] = len(context.split())
            gen_details["num_citations"] = len(citations)
            
            try:
                if on_token is not None:
                    answer_chunks: list[str] = []
                    try:
                        async for chunk in provider.chat_stream(messages):
                            answer_chunks.append(chunk)
                            on_token(chunk)
                    except NotImplementedError:
                        answer = await provider.chat(messages)
                        on_token(answer)
                        gen_details["streaming"] = False
                    else:
                        answer = "".join(answer_chunks)
                        gen_details["streaming"] = True
                else:
                    answer = await provider.chat(messages)
                gen_details["status"] = "success"
            except ProviderError as exc:
                answer = f"Provider error: {exc}"
                gen_details["status"] = "error"
                gen_details["error"] = str(exc)

        record_step(self._make_step("generation", gen_start, gen_start_time, gen_details))

        # Step 5: Self-reflection (optional retry)
        if on_stage:
            on_stage("reflection")
        reflection_start_time = datetime.utcnow()
        reflection_start = time.perf_counter()
        reflection_result = self._reflector.reflect(
            answer=answer,
            query=query,
            chunks=chunks,
            context_used=context,
        )
        reflection_details = {
            "quality": reflection_result.quality.value,
            "confidence": reflection_result.confidence,
            "issues": reflection_result.issues,
            "suggestions": reflection_result.suggestions,
            "should_retry": reflection_result.should_retry,
        }
        record_step(self._make_step("reflection", reflection_start, reflection_start_time, reflection_details))

        if reflection_result.should_retry and provider is not None:
            retry_top_k = max(6, int(plan.steps[0].dense_k * 1.5)) if plan.steps else 6
            if on_stage:
                on_stage("retrieval_retry")
            retry_retrieval_start_time = datetime.utcnow()
            retry_retrieval_start = time.perf_counter()
            retry_evidence = self._gatherer.gather(query, top_k=retry_top_k, document_ids=document_ids)
            chunks = retry_evidence.chunks
            record_step(
                self._make_step(
                    "retrieval_retry",
                    retry_retrieval_start,
                    retry_retrieval_start_time,
                    {
                        "top_k": retry_top_k,
                        "chunks_found": len(chunks),
                    },
                )
            )

            if on_stage:
                on_stage("compression_retry")
            retry_compression_start_time = datetime.utcnow()
            retry_compression_start = time.perf_counter()
            if use_compression and chunks:
                compressed = self._compressor.compress(
                    chunks,
                    query=query,
                    strategy="extractive",
                )
                context = compressed.text
                citations = compressed.citations
                compression_details = {
                    "enabled": True,
                    "chunks_used": compressed.chunks_used,
                    "chunks_total": compressed.chunks_total,
                    "estimated_tokens": compressed.estimated_tokens,
                }
                compression_status = "completed"
            else:
                context, citations = self._compressor.format_with_citations(chunks)
                compression_details = {
                    "enabled": False,
                    "chunks_used": len(chunks),
                    "chunks_total": len(chunks),
                }
                compression_status = "skipped"
            record_step(
                self._make_step(
                    "compression_retry",
                    retry_compression_start,
                    retry_compression_start_time,
                    compression_details,
                    status=compression_status,
                )
            )

            if on_stage:
                on_stage("generation_retry")
            retry_gen_start_time = datetime.utcnow()
            retry_gen_start = time.perf_counter()
            gen_details = {
                "provider": getattr(provider, "base_url", "unknown"),
                "model": getattr(provider, "default_model", "unknown"),
                "context_tokens": len(context.split()),
                "num_citations": len(citations),
                "retry": True,
            }
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
            try:
                if on_token is not None:
                    answer_chunks = []
                    try:
                        async for chunk in provider.chat_stream(messages):
                            answer_chunks.append(chunk)
                            on_token(chunk)
                    except NotImplementedError:
                        answer = await provider.chat(messages)
                        on_token(answer)
                        gen_details["streaming"] = False
                    else:
                        answer = "".join(answer_chunks)
                        gen_details["streaming"] = True
                else:
                    answer = await provider.chat(messages)
                gen_details["status"] = "success"
            except ProviderError as exc:
                answer = f"Provider error: {exc}"
                gen_details["status"] = "error"
                gen_details["error"] = str(exc)
            record_step(self._make_step("generation_retry", retry_gen_start, retry_gen_start_time, gen_details))

        # Calculate final metrics
        total_tokens = len(context.split()) if context else 0
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
                "cache_hit": False,
                "answer_quality": reflection_result.quality.value if reflection_result else "unknown",
                "answer_confidence": reflection_result.confidence if reflection_result else 0.0,
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
                "started_at": s.started_at.isoformat(),
                "completed_at": s.completed_at.isoformat(),
            }
            for s in pipeline_steps
        ]

        result = {
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
            "sources": citations,  # New: structured citation info
            "trace_id": trace.id,
            "metrics": {
                "chunks": len(chunks),
                "coverage": coverage,
                "tokens": total_tokens,
                "duration_ms": total_duration_ms,
                "query_type": str(query_type),
                "cache_hit": False,
                "answer_quality": reflection_result.quality.value if reflection_result else "unknown",
                "answer_confidence": reflection_result.confidence if reflection_result else 0.0,
            },
            "steps": steps_out,
        }
        cacheable = {**result, "steps": [s for s in steps_out if s["name"] != "cache"]}
        cache_manager.queries.set(query, cacheable, cache_hash)
        return result
