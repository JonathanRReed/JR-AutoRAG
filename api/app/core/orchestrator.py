"""Agentic orchestrator with iterative retrieval and self-correction.

This module implements a SOTA Auto-RAG pipeline with:
- Adaptive retrieval gating (Self-RAG style)
- CRAG-style retrieval quality evaluation
- Iterative retrieve-refine loops with marginal gain stopping
- 10/10 audit-ready citation enforcement
- Self-reflection and answer quality assessment
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from typing import Any, cast
from collections.abc import Callable

from ..schemas.config import AppConfig
from .cache import get_cache_manager
from .gatherer import Gatherer, EvidenceChunk
from .planner import Planner
from .providers import LLMProvider, ProviderError, ProviderFactory
from .retrieval import RetrievalEngine, HybridRetrievalEngine
from .reflection import SelfReflector
from .telemetry import PipelineStep, TelemetryStore
from .compression import ContextCompressor, CompressedContext
from .prompt_guard import CITATION_POLICY_PROMPT
# New agentic components
from .retrieval_evaluator import RetrievalEvaluator, RetrievalVerdict
from .adaptive_gate import AdaptiveGate, GateDecision
# Confidence monitoring
from .uncertainty_monitor import UncertaintyMonitor
# SOTA enhancements
from .flare import FLAREGenerator, FLAREConfig
from .hallucination_firewall import HallucinationFirewall
from .evidence_contract import EvidenceContract
# Advanced retrieval modes
from .hierarchy import HierarchyBuilder, HierarchicalRetriever, DocumentTree
from .graph_rag import GraphRAG
from .learned_router import LearnedRouter, RouteDecision
from .conflict_detector import ConflictDetector
from .budget_planner import BudgetPlanner, BudgetClass
from .decision_logger import get_decision_logger
from .ragas_eval import RAGASEvaluator, InvocationEvaluator
# Web search disabled for offline-only operation
# from .web_search import WebSearch, get_web_search
from .smart_planner import compute_marginal_gain


class Orchestrator:
    """Agentic RAG orchestrator with iterative retrieval and self-correction.
    
    Key SOTA features:
    - Adaptive gating: Decides if retrieval is needed at all
    - CRAG evaluation: Assesses context quality before generation
    - Iterative retrieval: Refines queries until evidence is sufficient
    - Web fallback: Falls back to web search when local retrieval fails
    - Self-reflection: Evaluates answer quality and triggers retry
    """
    
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
        # New agentic components
        self._retrieval_evaluator = RetrievalEvaluator()
        self._adaptive_gate = AdaptiveGate()
        self._uncertainty_monitor = UncertaintyMonitor()
        # SOTA enhancements
        self._flare_generator = FLAREGenerator(
            FLAREConfig(
                confidence_threshold=0.3,
                max_retrievals=3,
                lookahead_tokens=50,
            ),
            monitor=self._uncertainty_monitor,
        )
        self._hallucination_firewall = HallucinationFirewall(strict_mode=False)
        self._evidence_contract = EvidenceContract(min_coverage=0.7)
        # Advanced retrieval modes
        self._hierarchy_builder = HierarchyBuilder()
        self._document_trees: dict[str, DocumentTree] = {}
        self._graph_rag: GraphRAG | None = None
        self._graph_ready = False
        self._hierarchy_ready = False
        self._learned_router = LearnedRouter()
        self._conflict_detector = ConflictDetector()
        self._budget_planner = BudgetPlanner()
        self._decision_logger = get_decision_logger()
        # Quality evaluation
        self._ragas_evaluator = RAGASEvaluator()
        self._invocation_evaluator = InvocationEvaluator()
        self._chunk_records: list[tuple[str, Any]] = []
        # Web search disabled for offline-only operation
        self._web_search = None

    def rebuild(self, config: AppConfig) -> None:
        self._config = config
        self._planner.rebuild(config)
        if config.provider:
            self._provider = self._providers.build(config.provider)
        if hasattr(self._planner, "set_provider"):
            self._planner.set_provider(self._provider)
        self._retrieval.build()
        if hasattr(self._retrieval, "get_document_trees"):
            try:
                self._document_trees = self._retrieval.get_document_trees()
                self._hierarchy_ready = bool(self._document_trees)
            except Exception:
                self._document_trees = {}
                self._hierarchy_ready = False
        if hasattr(self._retrieval, "get_chunk_records"):
            try:
                self._chunk_records = self._retrieval.get_chunk_records()
            except Exception:
                self._chunk_records = []
        # Update compressor with config settings
        self._compressor = ContextCompressor(
            max_tokens=config.retrieval.max_context_tokens,
        )
        if not getattr(config.retrieval, "graph", False):
            self._graph_rag = None
            self._graph_ready = False
        # Web search disabled for offline-only operation
        # self._web_search = get_web_search()

    def set_planner(self, planner: Planner) -> None:
        self._planner = planner
    
    async def _retrieve_with_raptor(
        self,
        query: str,
        document_ids: list[str] | None,
        base_chunks: list,
    ) -> list:
        """RAPTOR-style retrieval: add overview summaries to leaf chunks.
        
        Retrieves high-level summaries from document hierarchy trees
        to provide context alongside granular chunks.
        """
        from .gatherer import EvidenceChunk
        enhanced_chunks = []
        
        for doc_id, tree in self._document_trees.items():
            if document_ids and doc_id not in document_ids:
                continue
            
            # Get overview from root/high-level nodes
            root = tree.get_node(tree.root_id)
            if root and root.summary:
                enhanced_chunks.append(EvidenceChunk(
                    id=f"{doc_id}_overview",
                    title=f"{root.title} (Overview)",
                    snippet=root.summary,
                    score=0.85,
                ))
            
            # Add mid-level summaries for context
            for child_id in root.children[:2] if root else []:
                child = tree.get_node(child_id)
                if child and child.summary:
                    enhanced_chunks.append(EvidenceChunk(
                        id=f"{doc_id}_{child_id}_summary",
                        title=f"{child.title} (Summary)",
                        snippet=child.summary,
                        score=0.75,
                    ))
        
        return enhanced_chunks
    
    async def _retrieve_with_graph(
        self,
        query: str,
        document_ids: list[str] | None,
    ) -> list:
        """GraphRAG-style retrieval using entity graph and community summaries.
        
        Uses knowledge graph to find related entities and their community
        summaries for global context.
        """
        from .gatherer import EvidenceChunk
        
        if not self._graph_rag or not self._graph_rag.graph:
            return []
        
        chunks = []
        
        # Find relevant entities via query
        relevant_entities = self._graph_rag.query_entities(query, top_k=5)
        
        # Get community summaries for matched entities
        for community in self._graph_rag.communities:
            entity_names = [e for e, _ in relevant_entities]
            if any(e in community.entities for e in entity_names):
                if community.summary:
                    chunks.append(EvidenceChunk(
                        id=f"community_{community.id}",
                        title=f"Topic: {', '.join(community.entities[:3])}",
                        snippet=community.summary,
                        score=0.8,
                    ))
        
        return chunks[:5]  # Limit community chunks

    async def _ensure_graph_context(self, force: bool = False) -> None:
        """Build GraphRAG context on demand."""
        if (
            not self._config
            or (not force and not getattr(self._config.retrieval, "graph", False))
            or self._graph_ready
        ):
            return
        if self._provider is None or not self._chunk_records:
            return
        try:
            graph_builder = GraphRAG()
            evidence_chunks: list[EvidenceChunk] = []
            # Limit to manageable number of chunks to control cost
            max_chunks = min(len(self._chunk_records), 400)
            for doc_id, chunk in self._chunk_records[:max_chunks]:
                tree = self._document_trees.get(doc_id)
                title = ""
                if tree:
                    root = tree.get_node(tree.root_id)
                    title = root.title if root else ""
                if not title:
                    title = f"Document {doc_id}"
                evidence_chunks.append(EvidenceChunk(
                    id=f"{doc_id}-{chunk.index}",
                    title=title,
                    snippet=chunk.text,
                    score=0.8,
                ))
            if not evidence_chunks:
                return
            await graph_builder.build_from_chunks(evidence_chunks, self._provider)
            graph_builder.detect_communities()
            await graph_builder.summarize_communities(self._provider)
            self._graph_rag = graph_builder
            self._graph_ready = True
        except Exception:
            self._graph_ready = False

    async def _ensure_hierarchy_context(self, force: bool = False) -> None:
        """Build hierarchical document trees if not provided by retriever."""
        if self._hierarchy_ready and not force:
            return
        if not self._chunk_records:
            return
        doc_texts: dict[str, list[str]] = {}
        for doc_id, chunk in self._chunk_records:
            text = getattr(chunk, "text", None) or getattr(chunk, "snippet", "")
            if not text:
                continue
            doc_texts.setdefault(doc_id, []).append(text)
        if not doc_texts:
            return
        try:
            trees: dict[str, DocumentTree] = {}
            for doc_id, parts in doc_texts.items():
                combined = "\n\n".join(parts)
                if not combined.strip():
                    continue
                tree = self._hierarchy_builder.build(
                    combined,
                    document_id=doc_id,
                    title=f"Document {doc_id}",
                )
                trees[doc_id] = tree
            if trees:
                self._document_trees = trees
                self._hierarchy_ready = True
        except Exception:
            self._hierarchy_ready = False

    def _split_chunk_identifier(self, chunk_id: str) -> tuple[str | None, str | None]:
        """Split chunk identifier into document and chunk tokens."""
        if "-" not in chunk_id:
            return None, None
        doc_part, _, chunk_part = chunk_id.rpartition("-")
        if not doc_part or not chunk_part:
            return None, None
        return doc_part, chunk_part

    def _multi_resolution_expand(
        self,
        chunks: list["EvidenceChunk"],
        document_ids: list[str] | None,
    ) -> list["EvidenceChunk"]:
        """Add parent/sibling context for top chunks."""
        if (
            not self._config
            or not getattr(self._config.retrieval, "multi_resolution", False)
            or not self._document_trees
        ):
            return []
        expanded: list[EvidenceChunk] = []
        seen_ids = {chunk.id for chunk in chunks}
        for chunk in chunks[:5]:
            doc_id, chunk_token = self._split_chunk_identifier(chunk.id)
            if not doc_id or not chunk_token:
                continue
            if document_ids and doc_id not in document_ids:
                continue
            tree = self._document_trees.get(doc_id)
            if not tree:
                continue
            retriever = HierarchicalRetriever(tree)
            context_chain = retriever.get_context_chain(chunk_token)
            for idx, ctx in enumerate(context_chain[:3]):
                context_id = f"{doc_id}_context_{chunk_token}_{idx}"
                if context_id in seen_ids:
                    continue
                expanded.append(EvidenceChunk(
                    id=context_id,
                    title=f"{chunk.title} Context",
                    snippet=ctx,
                    score=chunk.score * 0.9,
                ))
                seen_ids.add(context_id)
            siblings = retriever.expand_with_siblings(chunk_token)
            for idx, sibling in enumerate(siblings[:2]):
                sibling_id = f"{doc_id}_sibling_{chunk_token}_{idx}"
                if sibling_id in seen_ids:
                    continue
                expanded.append(EvidenceChunk(
                    id=sibling_id,
                    title="Related Section",
                    snippet=sibling,
                    score=chunk.score * 0.85,
                ))
                seen_ids.add(sibling_id)
        return expanded

    async def _run_uncertainty_monitored_generation(
        self,
        *,
        provider: LLMProvider,
        system_prompt: str,
        user_prompt: str,
        retriever: HybridRetrievalEngine,
        document_ids: list[str] | None,
        context_text: str,
        query: str,
        strict_instruction: str,
        continue_instruction: str,
        gen_details: dict[str, Any],
    ) -> str:
        """Generate once, then inspect confidence; if low, trigger FLARE refinement."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        base_answer = await provider.chat(messages)

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", base_answer)
            if s.strip()
        ]
        triggers: list[dict[str, Any]] = []
        threshold = getattr(self._uncertainty_monitor, "threshold", 0.35)
        def _maybe_token_stats(text: str) -> dict[str, float | None]:
            if provider and hasattr(provider, "get_token_stats") and callable(getattr(provider, "get_token_stats")):
                try:
                    stats = provider.get_token_stats(text)  # type: ignore[attr-defined]
                    if isinstance(stats, dict):
                        return {
                            "avg_logprob": stats.get("avg_logprob"),
                            "entropy": stats.get("entropy"),
                            "logit_margin": stats.get("logit_margin"),
                        }
                except Exception:
                    return {}
            return {}

        for idx, sentence in enumerate(sentences):
            stats = _maybe_token_stats(sentence)
            signal = self._uncertainty_monitor.estimate(
                sentence,
                avg_logprob=stats.get("avg_logprob"),
                entropy=stats.get("entropy"),
                logit_margin=stats.get("logit_margin"),
            )
            if signal.aggregate < threshold:
                triggers.append(
                    {
                        "index": idx,
                        "confidence": round(signal.aggregate, 3),
                        "preview": sentence[:160],
                    }
                )
                if len(triggers) >= self._flare_generator.config.max_retrievals:
                    break

        if not triggers:
            gen_details["uncertainty_triggers"] = 0
            return base_answer

        gen_details["uncertainty_triggers"] = triggers

        try:
            flare_result = await self._flare_generator.generate_with_flare(
                query=query,
                initial_context=context_text,
                provider=provider,
                retriever=retriever,
                document_ids=document_ids,
                system_prompt=system_prompt,
                answer_instruction=strict_instruction,
                continue_instruction=continue_instruction,
            )
        except Exception as exc:
            gen_details["uncertainty_fallback_error"] = str(exc)
            return base_answer

        gen_details["mode"] = "uncertainty_flare"
        gen_details["flare_retrievals"] = flare_result.total_retrievals
        gen_details["flare_chunks_used"] = flare_result.total_chunks_used
        gen_details["flare_steps"] = len(flare_result.steps)
        return flare_result.answer

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

        def dedupe_chunks(chunk_list: list) -> list:
            """Deduplicate chunks by ID, keeping the highest score."""
            seen: dict[str, Any] = {}
            for chunk in chunk_list:
                chunk_id = getattr(chunk, "id", str(id(chunk)))
                current_best = seen.get(chunk_id)
                if current_best is None or getattr(chunk, "score", 0) > getattr(current_best, "score", 0):
                    seen[chunk_id] = chunk
            return list(seen.values())

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
                "iterative": getattr(plan, "iterative", False),
                "max_iterations": getattr(plan, "max_iterations", 1),
            },
        ))
        routing_params = dict(getattr(plan, "routing_params", {}) or {})

        # Step 1.5: Adaptive Gating (new agentic step)
        if on_stage:
            on_stage("gating")
        gating_start_time = datetime.utcnow()
        gating_start = time.perf_counter()
        
        gate_result = await self._adaptive_gate.should_retrieve(query, self._provider)
        
        gating_details = {
            "decision": gate_result.decision.value,
            "confidence": gate_result.confidence,
            "reasoning": gate_result.reasoning,
        }
        
        # Handle no-retrieval case (LLM can answer directly)
        if gate_result.decision == GateDecision.NO_RETRIEVAL:
            record_step(self._make_step("gating", gating_start, gating_start_time, gating_details))
            # Generate direct answer without retrieval
            return await self._generate_direct_answer(
                query, pipeline_start, pipeline_steps, cache_hash,
                on_step, on_token, on_stage, record_step
            )
        
        # Handle clarification case
        if gate_result.decision == GateDecision.CLARIFY_FIRST:
            gating_details["clarification"] = gate_result.clarification_question
            record_step(self._make_step("gating", gating_start, gating_start_time, gating_details))
            # Return clarification request
            return self._build_clarification_response(
                query, gate_result.clarification_question, pipeline_start, pipeline_steps
            )
        
        # Update max iterations based on gating decision
        max_iterations = getattr(plan, "max_iterations", 1)
        if gate_result.decision == GateDecision.ITERATIVE_RETRIEVAL:
            max_iterations = max(max_iterations, gate_result.suggested_iterations)
        
        gating_details["max_iterations"] = max_iterations
        record_step(self._make_step("gating", gating_start, gating_start_time, gating_details))
        
        # Log gating decision for training data collection
        self._decision_logger.log_gate_decision(
            query=query,
            decision=gate_result.decision.value,
            confidence=gate_result.confidence,
            reasoning=gate_result.reasoning,
        )
        
        # Step 1.6: Learned Router for advanced retrieval strategy
        if on_stage:
            on_stage("routing")
        routing_start_time = datetime.utcnow()
        routing_start = time.perf_counter()
        
        learned_route = self._learned_router.route(query)
        
        # Budget-aware planning
        query_complexity = self._budget_planner.estimate_query_complexity(query)
        budget_plan = self._budget_planner.plan(
            budget_class=BudgetClass.STANDARD,
            query_complexity=query_complexity,
        )
        
        # Determine if RAPTOR or Graph modes should be used
        cfg_retrieval = getattr(self._config, "retrieval", None)
        cfg_use_raptor = bool(getattr(cfg_retrieval, "raptor", False))
        cfg_use_graph = bool(getattr(cfg_retrieval, "graph", False))
        base_use_raptor = (
            learned_route.decision == RouteDecision.RAPTOR
            or cfg_use_raptor
            or (budget_plan.use_raptor and bool(self._document_trees))
        )
        base_use_graph = (
            learned_route.decision == RouteDecision.GRAPH
            or cfg_use_graph
            or (budget_plan.use_graph and self._graph_rag is not None)
        )
        use_raptor = base_use_raptor
        use_graph = base_use_graph
        
        routing_details = {
            "learned_route": {
                "decision": learned_route.decision.value,
                "confidence": learned_route.confidence,
                "suggested_k": learned_route.suggested_k,
                "use_rerank": learned_route.use_rerank,
                "max_iterations": learned_route.max_iterations,
            },
            "budget_plan": budget_plan.to_dict(),
            "query_complexity": round(query_complexity, 3),
            "use_raptor": use_raptor,
            "use_graph": use_graph,
            "use_colbert_plan": budget_plan.use_colbert,
        }
        record_step(self._make_step("routing", routing_start, routing_start_time, routing_details))
        
        # Log routing decision
        self._decision_logger.log_route_decision(
            query=query,
            decision=learned_route.decision.value,
            features=learned_route.features.to_dict(),
            suggested_k=learned_route.suggested_k,
            use_rerank=learned_route.use_rerank,
            max_iterations=learned_route.max_iterations,
        )
        
        # Adjust max_iterations with learned routing guidance
        max_iterations = max(max_iterations, learned_route.max_iterations)
        # Override max_iterations from budget plan if tighter
        max_iterations = min(max_iterations, budget_plan.max_iterations)

        # Adjust retrieval k per learned router suggestion
        for step in getattr(plan, "steps", []):
            if hasattr(step, "dense_k"):
                step.dense_k = max(step.dense_k, learned_route.suggested_k or step.dense_k)
            if hasattr(step, "sparse_k"):
                step.sparse_k = max(step.sparse_k, learned_route.suggested_k or step.sparse_k)

        await self._ensure_hierarchy_context(force=base_use_raptor)
        await self._ensure_graph_context(force=base_use_graph)
        # Enable high precision mode if budget allows
        if isinstance(self._retrieval, HybridRetrievalEngine):
            self._retrieval.set_precision_mode(budget_plan.use_colbert)
            # Enable/disable reranker based on learned route + budget
            rerank_enabled = learned_route.use_rerank or budget_plan.use_rerank
            self._retrieval.set_rerank_mode(rerank_enabled)
            routing_details["rerank_enabled"] = rerank_enabled
            routing_details["use_raptor"] = use_raptor
            routing_details["use_graph"] = use_graph
            routing_details["use_colbert"] = budget_plan.use_colbert

        # Step 2: Iterative Gatherer with CRAG evaluation
        if on_stage:
            on_stage("gatherer")
        gatherer_start_time = datetime.utcnow()
        gatherer_start = time.perf_counter()
        
        all_chunks = []
        gatherer_details: dict[str, Any] = {
            "sub_queries": [], 
            "literal_hits": 0,
            "iterations": [],
            "total_iterations": 0,
            "raptor_total": 0,
            "graph_total": 0,
            "raptor_ready": bool(self._document_trees),
            "graph_ready": self._graph_rag is not None,
        }
        total_raptor_chunks = 0
        total_graph_chunks = 0
        embedding_cache_hits = 0
        embedding_cache_misses = 0
        
        # Iterative retrieval loop
        iteration = 0
        accumulated_chunks = []
        current_query = query
        retrieval_verdict = None
        plan_queries = [s.query for s in plan.steps if getattr(s, "query", None)]
        
        target_coverage = getattr(plan, "coverage_target", 0.75)
        final_coverage_ratio = 0.0
        # Corrective policy knobs (kept conservative to bound cost)
        policy_config = {
            "incorrect_widen_factor": 1.5,
            "max_slot_fill_queries": 2,
            "slot_fill_top_k": 3,
            "max_clarification_queries": 2,
            "clarification_top_k": 4,
        }

        while iteration < max_iterations:
            iteration_start = time.perf_counter()
            iteration_chunks = []
            iteration_details = {"iteration": iteration + 1, "query": current_query, "sub_queries": []}
            
            # Execute retrieval for all plan steps
            for step in plan.steps:
                # Use refined query for iterations > 0
                step_query = step.query if iteration == 0 else current_query
                sub_start = time.perf_counter()
                step_evidence = self._gatherer.gather(
                    step_query,
                    top_k=step.dense_k,
                    document_ids=document_ids,
                    routing_params=routing_params,
                )
                iteration_chunks.extend(step_evidence.chunks)
                
                embedding_cache = step_evidence.cache_info.get("embedding_cache")
                literal_hits = step_evidence.cache_info.get("literal_hits", 0)
                if embedding_cache == "hit":
                    embedding_cache_hits += 1
                elif embedding_cache == "miss":
                    embedding_cache_misses += 1
                if isinstance(literal_hits, int):
                    gatherer_details["literal_hits"] += literal_hits
                
                sub_query_info = {
                    "query": step_query,
                    "top_k": step.dense_k,
                    "chunks_found": len(step_evidence.chunks),
                    "duration_ms": round((time.perf_counter() - sub_start) * 1000, 2),
                    "embedding_cache": embedding_cache,
                    "literal_hits": literal_hits,
                }
                gatherer_details["sub_queries"].append(sub_query_info)
                iteration_details["sub_queries"].append(sub_query_info)
            
            # CRAG: Evaluate retrieval quality
            eval_result = await self._retrieval_evaluator.evaluate(
                current_query, iteration_chunks, self._provider
            )
            retrieval_verdict = eval_result.verdict
            iteration_details["verdict"] = eval_result.verdict.value
            iteration_details["verdict_confidence"] = eval_result.confidence
            relevance_stats = self._retrieval_evaluator.filter_relevant_sentences(
                current_query, iteration_chunks
            )
            if relevance_stats.get("trimmed_chunks"):
                iteration_details["relevance_filter"] = relevance_stats
            
            # Also check coverage for more granular assessment
            coverage_verdict, coverage_ratio, missing_aspects = self._retrieval_evaluator.evaluate_coverage(
                current_query, iteration_chunks
            )
            iteration_details["coverage_ratio"] = coverage_ratio
            iteration_details["missing_aspects"] = missing_aspects[:3]  # Limit for logging
            final_coverage_ratio = coverage_ratio
            plan_coverage_ratio, missing_plan_queries = self._retrieval_evaluator.evaluate_plan_coverage(
                plan_queries, iteration_chunks
            )
            plan_coverage_target = min(0.85, max(0.6, target_coverage))
            iteration_details["plan_coverage_ratio"] = round(plan_coverage_ratio, 3)
            iteration_details["plan_coverage_target"] = plan_coverage_target
            iteration_details["missing_plan_queries"] = missing_plan_queries[:3]
            plan_coverage_sufficient = plan_coverage_ratio >= plan_coverage_target
            if not plan_coverage_sufficient:
                iteration_details["plan_coverage_gap"] = round(
                    max(0.0, plan_coverage_target - plan_coverage_ratio), 3
                )
            coverage_sufficient = coverage_ratio >= target_coverage and plan_coverage_sufficient
            if not coverage_sufficient:
                iteration_details["coverage_gap"] = round(max(0.0, target_coverage - coverage_ratio), 3)
            
            # Handle INCORRECT verdict - widen search or try slot-fill
            if eval_result.verdict == RetrievalVerdict.INCORRECT:
                iteration_details["corrective_action"] = "widen_search"
                widen_factor = policy_config["incorrect_widen_factor"]
                # Increase k by configured factor for next iteration
                for step in plan.steps:
                    step.dense_k = int(step.dense_k * widen_factor)
                    step.sparse_k = int(getattr(step, 'sparse_k', step.dense_k) * widen_factor)
                iteration_details["new_dense_k"] = plan.steps[0].dense_k if plan.steps else 0
                # Activate advanced modes after incorrect verdict
                if not use_raptor and self._document_trees:
                    use_raptor = True
                    iteration_details["policy_enable_raptor"] = True
                if not use_graph and self._graph_rag is not None:
                    use_graph = True
                    iteration_details["policy_enable_graph"] = True
                # Opportunistically pull RAPTOR/Graph context immediately
                if self._document_trees and iteration_details.get("raptor_chunks_added") is None:
                    raptor_chunks = await self._retrieve_with_raptor(
                        current_query, document_ids, iteration_chunks
                    )
                    if raptor_chunks:
                        iteration_chunks.extend(raptor_chunks)
                        iteration_details["raptor_chunks_added"] = len(raptor_chunks)
                if self._graph_rag is not None and iteration_details.get("graph_chunks_added") is None:
                    graph_chunks = await self._retrieve_with_graph(current_query, document_ids)
                    if graph_chunks:
                        iteration_chunks.extend(graph_chunks)
                        iteration_details["graph_chunks_added"] = len(graph_chunks)

                # Web fallback only if available (fixed null check)
                if self._web_search is not None and hasattr(self._web_search, 'available') and self._web_search.available:
                    web_start = time.perf_counter()
                    web_results = await self._web_search.search(current_query, max_results=3)
                    if web_results:
                        iteration_details["web_fallback"] = True
                        iteration_details["web_results"] = len(web_results)
                        # Convert web results to chunk-like format
                        for wr in web_results:
                            from .gatherer import EvidenceChunk
                            web_chunk = EvidenceChunk(
                                id=f"web_{hash(wr.url) % 10000}",
                                title=wr.title,
                                snippet=wr.snippet,
                                score=0.5,
                            )
                            iteration_chunks.append(web_chunk)
                    iteration_details["web_duration_ms"] = round((time.perf_counter() - web_start) * 1000, 2)
            
            # Handle LOW_COVERAGE verdict - generate slot-fill queries
            if coverage_verdict == RetrievalVerdict.LOW_COVERAGE:
                slot_fill_queries = self._retrieval_evaluator.generate_slot_fill_queries(
                    current_query, iteration_chunks, self._provider
                )
                iteration_details["corrective_action"] = "slot_fill"
                iteration_details["slot_fill_queries"] = slot_fill_queries
                
                # Execute slot-fill queries for targeted retrieval
                for sq in slot_fill_queries[: policy_config["max_slot_fill_queries"]]:
                    sub_evidence = self._gatherer.gather(
                        sq,
                        top_k=policy_config["slot_fill_top_k"],
                        document_ids=document_ids,
                        routing_params=routing_params,
                    )
                    iteration_chunks.extend(sub_evidence.chunks)
                    iteration_details["slot_fill_chunks_added"] = iteration_details.get("slot_fill_chunks_added", 0) + len(sub_evidence.chunks)
            
            # Handle AMBIGUOUS verdict with knowledge strip extraction
            if eval_result.verdict == RetrievalVerdict.AMBIGUOUS:
                knowledge_strips = self._retrieval_evaluator.extract_knowledge_strips(
                    iteration_chunks, current_query
                )
                iteration_details["knowledge_strips_extracted"] = len(knowledge_strips)
                clarification_queries = self._retrieval_evaluator.generate_clarification_queries(
                    current_query, iteration_chunks
                )
                if clarification_queries:
                    iteration_details["clarification_queries"] = clarification_queries
                    for cq in clarification_queries[: policy_config["max_clarification_queries"]]:
                        clar_evidence = self._gatherer.gather(
                            cq,
                            top_k=policy_config["clarification_top_k"],
                            document_ids=document_ids,
                            routing_params=routing_params,
                        )
                        iteration_chunks.extend(clar_evidence.chunks)
                        iteration_details.setdefault("clarification_chunks_added", 0)
                        iteration_details["clarification_chunks_added"] += len(clar_evidence.chunks)
                
                # Try decomposing the query for targeted retrieval
                if hasattr(self._planner, 'decompose_query'):
                    try:
                        sub_queries = await self._planner.decompose_query(current_query)
                        for sq in sub_queries[:2]:  # Limit to 2 sub-queries
                            sub_evidence = self._gatherer.gather(
                                sq,
                                top_k=3,
                                document_ids=document_ids,
                                routing_params=routing_params,
                            )
                            iteration_chunks.extend(sub_evidence.chunks)
                        iteration_details["decomposed_queries"] = len(sub_queries)
                    except Exception:
                        pass  # Planner may not support decomposition
            
            # RAPTOR: Add hierarchical overview chunks if enabled on first hop
            if use_raptor and self._document_trees and iteration == 0:
                raptor_chunks = await self._retrieve_with_raptor(
                    current_query, document_ids, iteration_chunks
                )
                if raptor_chunks:
                    iteration_chunks.extend(raptor_chunks)
                    iteration_details["raptor_chunks_added"] = len(raptor_chunks)
                    iteration_details["raptor_enabled"] = True
                    total_raptor_chunks += len(raptor_chunks)
            
            # GraphRAG: Add community summaries if enabled on first hop
            if use_graph and self._graph_rag is not None and iteration == 0:
                graph_chunks = await self._retrieve_with_graph(current_query, document_ids)
                if graph_chunks:
                    iteration_chunks.extend(graph_chunks)
                    iteration_details["graph_chunks_added"] = len(graph_chunks)
                    iteration_details["graph_enabled"] = True
                    total_graph_chunks += len(graph_chunks)
            
            # Log retrieval verdict for training data
            self._decision_logger.log_retrieval_verdict(
                query=current_query,
                verdict=eval_result.verdict.value,
                confidence=eval_result.confidence,
                chunks_count=len(iteration_chunks),
                iteration=iteration + 1,
                corrective_action=iteration_details.get("corrective_action"),
                coverage_ratio=coverage_ratio,
                missing_aspects=missing_aspects[:5],
            )
            
            # Add to accumulated chunks
            for chunk in iteration_chunks:
                accumulated_chunks.append(chunk)
            
            iteration_details["chunks_this_iteration"] = len(iteration_chunks)
            iteration_details["duration_ms"] = round((time.perf_counter() - iteration_start) * 1000, 2)
            gatherer_details["iterations"].append(iteration_details)
            
            # Check stopping criteria
            if iteration > 0 and len(accumulated_chunks) > 0:
                # Compute marginal gain
                prev_chunk_dicts = [{"text": c.snippet, "score": c.score} for c in accumulated_chunks[:-len(iteration_chunks)]]
                new_chunk_dicts = [{"text": c.snippet, "score": c.score} for c in iteration_chunks]
                marginal_gain = compute_marginal_gain(prev_chunk_dicts, new_chunk_dicts)
                
                iteration_details["marginal_gain"] = marginal_gain
                stop_threshold = getattr(plan, "stop_threshold", 0.1)
                
                if marginal_gain < stop_threshold and coverage_sufficient:
                    iteration_details["stopped_reason"] = f"marginal_gain ({marginal_gain:.3f}) < threshold ({stop_threshold})"
                    break
            
            # Check if verdict suggests we should stop or refine
            if eval_result.verdict == RetrievalVerdict.CORRECT and coverage_sufficient:
                iteration_details["stopped_reason"] = "verdict_correct"
                break
            
            # Refine query for next iteration if needed
            if eval_result.suggested_query and iteration < max_iterations - 1:
                current_query = eval_result.suggested_query
                iteration_details["refined_query"] = current_query
            
            iteration += 1
        
        all_chunks = accumulated_chunks
        gatherer_details["total_iterations"] = iteration + 1
        gatherer_details["final_verdict"] = retrieval_verdict.value if retrieval_verdict else "unknown"
        gatherer_details["embedding_cache_hits"] = embedding_cache_hits
        gatherer_details["embedding_cache_misses"] = embedding_cache_misses
        gatherer_details["raptor_total"] = total_raptor_chunks
        gatherer_details["graph_total"] = total_graph_chunks
        record_step(self._make_step("gatherer", gatherer_start, gatherer_start_time, gatherer_details))

        # Step 3: Retrieval (aggregation + dedupe)
        if on_stage:
            on_stage("retrieval")
        retrieval_start_time = datetime.utcnow()
        retrieval_start = time.perf_counter()
        retrieval_details: dict[str, Any] = {"sub_queries": gatherer_details["sub_queries"]}
        precision_stats: dict[str, Any] = {}
        colbert_enabled = False
        rerank_enabled = False

        # Deduplicate chunks by ID, keeping highest score
        seen: dict[str, Any] = {}
        for chunk in all_chunks:
            if chunk.id not in seen or chunk.score > seen[chunk.id].score:
                seen[chunk.id] = chunk
        
        chunks = list(seen.values())
        # Sort by score descending
        chunks.sort(key=lambda c: c.score, reverse=True)
        
        if not chunks:
            evidence = self._gatherer.gather(
                query,
                top_k=3,
                document_ids=document_ids,
                routing_params=routing_params,
            )
            chunks = evidence.chunks

        # Multi-resolution expansion (parent/child context)
        extra_context: list = []
        if getattr(self._config.retrieval, "multi_resolution", False):
            extra_context = self._multi_resolution_expand(chunks, document_ids)
            if extra_context:
                chunks.extend(extra_context)
                chunks.sort(key=lambda c: c.score, reverse=True)

        if document_ids:
            retrieval_details["document_filter"] = document_ids
        retrieval_details["total_chunks"] = len(chunks)
        retrieval_details["unique_sources"] = len({c.title for c in chunks})
        retrieval_details["embedding_cache_hits"] = embedding_cache_hits
        retrieval_details["embedding_cache_misses"] = embedding_cache_misses
        if extra_context:
            retrieval_details["multi_resolution_expanded"] = len(extra_context)
        if total_raptor_chunks:
            retrieval_details["raptor_chunks_total"] = total_raptor_chunks
        if total_graph_chunks:
            retrieval_details["graph_chunks_total"] = total_graph_chunks
        if hasattr(self._retrieval, "model_status"):
            retrieval_details.update(self._retrieval.model_status())
        if isinstance(self._retrieval, HybridRetrievalEngine):
            try:
                retrieval_details["precision_stats"] = self._retrieval.precision_stats()
                retrieval_details["colbert_enabled"] = self._retrieval.colbert_enabled()
            except Exception:
                retrieval_details["precision_stats"] = {"applied": False}
            precision_stats = retrieval_details.get("precision_stats", {})
            colbert_enabled = retrieval_details.get("colbert_enabled", False)
            rerank_enabled = retrieval_details.get("rerank_enabled", rerank_enabled)
        record_step(self._make_step("retrieval", retrieval_start, retrieval_start_time, retrieval_details))

        compression_enabled = bool(self._config and self._config.retrieval.compression)

        def run_compression_step(step_name: str, chunk_list: list["EvidenceChunk"]) -> tuple[str, list[dict]]:
            if on_stage:
                on_stage(step_name)
            compression_start_time = datetime.utcnow()
            compression_start = time.perf_counter()
            if compression_enabled and chunk_list:
                compressed = self._compressor.compress(
                    chunk_list,
                    query=query,
                    strategy="extractive",
                )
                context_text = compressed.text
                citation_payload = compressed.citations
                compression_details = {
                    "enabled": True,
                    "chunks_used": compressed.chunks_used,
                    "chunks_total": compressed.chunks_total,
                    "estimated_tokens": compressed.estimated_tokens,
                }
                compression_status = "completed"
            else:
                context_text, citation_payload = self._compressor.format_with_citations(chunk_list)
                compression_details = {
                    "enabled": False,
                    "chunks_used": len(chunk_list),
                    "chunks_total": len(chunk_list),
                }
                compression_status = "skipped"
            record_step(
                self._make_step(
                    step_name,
                    compression_start,
                    compression_start_time,
                    compression_details,
                    status=compression_status,
                )
            )
            return context_text, citation_payload

        enable_flare = bool(
            self._config
            and getattr(self._config.retrieval, "flare_generation", False)
            and isinstance(self._retrieval, HybridRetrievalEngine)
        )
        retriever_for_flare = cast(HybridRetrievalEngine, self._retrieval) if enable_flare else None
        enforce_evidence_contract = bool(
            self._config and getattr(self._config.retrieval, "enforce_evidence_contract", False)
        )
        max_evidence_contract_retries = 2

        async def run_generation_step(
            step_name: str,
            context_text: str,
            citation_payload: list[dict],
            conflict_text: str,
            allow_flare: bool = True,
            extra_details: dict[str, Any] | None = None,
        ) -> tuple[str, dict]:
            if on_stage:
                on_stage(step_name)
            gen_start_time = datetime.utcnow()
            gen_start = time.perf_counter()
            provider = self._provider
            gen_details: dict[str, Any] = {
                "provider": getattr(provider, "base_url", "unknown") if provider else "none",
                "model": getattr(provider, "default_model", "unknown") if provider else None,
                "context_tokens": len(context_text.split()) if context_text else 0,
                "num_citations": len(citation_payload),
                "mode": "standard",
            }
            if extra_details:
                gen_details.update(extra_details)
            strict_instruction = (
                "Follow an evidence-first contract: (1) list the exact supporting quotes with citations "
                "under a short 'Evidence' preface, (2) then write the answer using only those quotes, "
                "and (3) ensure every factual claim includes an in-line citation. "
                "Never invent dates or numbers; if unsupported, state 'Unknown' or request more retrieval."
            )
            continue_instruction = (
                "Provide just the next sentence of the answer, grounded in the already listed evidence quotes, "
                "and include the supporting quote with citation. Do not add claims without evidence."
            )

            if provider is None:
                answer = (
                    f"(No provider configured.) Context summary:\n{context_text}"
                    if context_text
                    else "No documents ingested yet."
                )
                gen_details["fallback"] = True
                gen_details["status"] = "skipped"
            else:
                system_prompt = f"""You are JR AutoRAG assistant, a precise enterprise RAG generator with STRICT citation requirements.

{CITATION_POLICY_PROMPT}

CRITICAL REMINDERS:
- First extract supporting spans, then compose the answer only from those spans
- Every factual claim MUST have a 10-25 word quote from the source
- Format: "<claim>." "<exact quoted text>" (Doc: Title, ChunkID: xxx)
- If a date, number, or timeline is NOT in the sources, write: "[No dated catalyst found in knowledge base]"
- End with "## References" section listing all sources
- End with "## Sources Used" section confirming only KB sources used
- NEVER invent data - better to say "Unknown" than guess{conflict_text}"""

                user_prompt = (
                    f"Context (use ONLY this for your answer):\n{context_text}\n\n"
                    f"Question: {query}\n\n{strict_instruction}"
                )

                try:
                    if allow_flare and enable_flare and retriever_for_flare is not None:
                        flare_kwargs = {
                            "query": query,
                            "initial_context": context_text,
                            "provider": provider,
                            "retriever": retriever_for_flare,
                            "document_ids": document_ids,
                            "system_prompt": system_prompt,
                            "answer_instruction": strict_instruction,
                            "continue_instruction": continue_instruction,
                        }
                        if on_token is not None:
                            flare_result = await self._flare_generator.generate_streaming_with_flare(
                                on_token=on_token,
                                **flare_kwargs,
                            )
                        else:
                            flare_result = await self._flare_generator.generate_with_flare(**flare_kwargs)
                        answer = flare_result.answer
                        gen_details["mode"] = "flare"
                        gen_details["status"] = "success"
                        gen_details["flare_retrievals"] = flare_result.total_retrievals
                        gen_details["flare_chunks_used"] = flare_result.total_chunks_used
                        gen_details["flare_steps"] = len(flare_result.steps)
                    else:
                        if enable_flare and retriever_for_flare is not None:
                            # Monitor streamed generation for uncertainty dips and trigger FLARE on demand
                            answer = await self._run_uncertainty_monitored_generation(
                                provider=provider,
                                system_prompt=system_prompt,
                                user_prompt=user_prompt,
                                retriever=retriever_for_flare,
                                document_ids=document_ids,
                                context_text=context_text,
                                query=query,
                                strict_instruction=strict_instruction,
                                continue_instruction=continue_instruction,
                                gen_details=gen_details,
                            )
                        else:
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ]
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

            record_step(self._make_step(step_name, gen_start, gen_start_time, gen_details))
            return answer, gen_details

        # Step 3: Context Compression (new!)
        context, citations = run_compression_step("compression", chunks)

        # Step 3.5: Conflict Detection
        if on_stage:
            on_stage("conflict_detection")
        conflict_start_time = datetime.utcnow()
        conflict_start = time.perf_counter()
        
        conflict_result = self._conflict_detector.detect(chunks)
        
        conflict_details = {
            "has_conflicts": conflict_result.has_conflicts,
            "conflict_count": len(conflict_result.conflicts),
            "resolution_strategy": conflict_result.resolution_strategy,
        }
        
        # Build conflict handling instruction for generation if conflicts found
        conflict_instruction = ""
        if conflict_result.has_conflicts:
            conflict_details["conflicts_summary"] = [
                f"{c1[:50]}... vs {c2[:50]}..." 
                for c1, c2, _ in conflict_result.conflicts[:3]
            ]
            conflict_instruction = """

CONFLICT HANDLING:
The retrieved evidence contains some conflicting information. When you encounter conflicts:
1. Acknowledge the conflicting viewpoints
2. Present both perspectives with their sources
3. State which source appears more authoritative if applicable
4. Do not fabricate a resolution - acknowledge uncertainty"""
        
        record_step(self._make_step("conflict_detection", conflict_start, conflict_start_time, conflict_details))

        # Step 4: Generation (with optional FLARE mid-generation retrieval)
        answer, gen_details = await run_generation_step(
            "generation",
            context,
            citations,
            conflict_instruction,
            allow_flare=True,
        )

        def run_firewall_step(
            step_name: str,
            answer_text: str,
            chunk_list: list["EvidenceChunk"],
        ) -> tuple[str, dict]:
            stage_label = "verification" if step_name == "verification" else "verification_retry"
            if on_stage:
                on_stage(stage_label)
            firewall_start_time = datetime.utcnow()
            firewall_start = time.perf_counter()
            firewall_result = self._hallucination_firewall.verify(
                answer=answer_text,
                chunks=chunk_list,
                query=query,
            )
            firewall_details = {
                "verified_claims": firewall_result.verified_claims,
                "total_claims": firewall_result.total_claims,
                "pass_rate": round(firewall_result.pass_rate, 3),
                "flagged_claims_count": len(firewall_result.flagged_claims),
                "meets_threshold": firewall_result.details.get("meets_threshold", True),
            }
            sanitized_answer = answer_text
            if self._hallucination_firewall.strict_mode:
                sanitized_answer = firewall_result.cleaned_answer
                firewall_details["answer_modified"] = sanitized_answer != firewall_result.original_answer
            record_step(self._make_step(step_name, firewall_start, firewall_start_time, firewall_details))
            return sanitized_answer, firewall_details

        # Step 4.5: Hallucination Firewall (SOTA enhancement)
        answer, firewall_details = run_firewall_step("verification", answer, chunks)

        # Step 4.5b: Firewall-driven corrective retry (lightweight)
        # If the pass rate is low, attempt one targeted widen-and-regenerate pass.
        try:
            firewall_pass_rate = firewall_details.get("pass_rate", 1.0)
        except Exception:
            firewall_pass_rate = 1.0

        if (
            firewall_pass_rate < 0.6
            and provider is not None
            and chunks
        ):
            # Widen retrieval slightly and regenerate once without FLARE to keep latency bounded
            if on_stage:
                on_stage("verification_retry_trigger")
            retry_firewall_start_time = datetime.utcnow()
            retry_firewall_start = time.perf_counter()
            widen_k = max(6, int(plan.steps[0].dense_k * 1.25)) if plan.steps else 6
            retry_evidence = self._gatherer.gather(
                query,
                top_k=widen_k,
                document_ids=document_ids,
                routing_params=routing_params,
            )
            chunks = dedupe_chunks(chunks + retry_evidence.chunks)
            retry_details = {
                "reason": "firewall_low_pass_rate",
                "previous_pass_rate": firewall_pass_rate,
                "widen_k": widen_k,
                "chunks_found": len(retry_evidence.chunks),
            }
            record_step(
                self._make_step(
                    "verification_retry_trigger",
                    retry_firewall_start,
                    retry_firewall_start_time,
                    retry_details,
                )
            )

            # Recompress with the new chunks
            context, citations = run_compression_step(
                "compression_verification_retry",
                chunks,
            )

            # Regenerate once (no FLARE to bound latency), then re-run firewall
            answer, _ = await run_generation_step(
                "generation_verification_retry",
                context,
                citations,
                conflict_instruction,
                allow_flare=False,
            )
            answer, firewall_details = run_firewall_step(
                "verification_retry",
                answer,
                chunks,
            )

        # Step 4.6: Evidence-first contract enforcement
        contract_result = None
        contract_checks = 0
        contract_retries = 0
        if enforce_evidence_contract:
            while True:
                contract_checks += 1
                contract_step_name = (
                    "evidence_contract"
                    if contract_checks == 1
                    else f"evidence_contract_retry_{contract_retries}"
                )
                if on_stage:
                    on_stage("evidence_contract" if contract_checks == 1 else "evidence_contract_retry")
                contract_start_time = datetime.utcnow()
                contract_start = time.perf_counter()
                contract_result = self._evidence_contract.verify_answer(answer, chunks)
                contract_details = {
                    "coverage_ratio": round(contract_result.coverage_ratio, 3),
                    "pass_threshold": contract_result.pass_threshold,
                    "verified_claims": len(contract_result.verified_claims),
                    "unsupported_claims": len(contract_result.unsupported_claims),
                    "suggested_retrievals": contract_result.suggested_retrievals[:3],
                }
                record_step(self._make_step(contract_step_name, contract_start, contract_start_time, contract_details))
                if contract_result.pass_threshold or contract_retries >= max_evidence_contract_retries:
                    break
                suggestions = [s for s in contract_result.suggested_retrievals if s][:3]
                if not suggestions:
                    break
                targeted_chunks: list = []
                for suggestion in suggestions:
                    try:
                        bundle = self._gatherer.gather(
                            suggestion,
                            top_k=4,
                            document_ids=document_ids,
                            routing_params=routing_params,
                        )
                        targeted_chunks.extend(bundle.chunks)
                    except Exception:
                        continue
                if not targeted_chunks:
                    break
                chunks.extend(targeted_chunks)
                chunks = dedupe_chunks(chunks)
                contract_retries += 1
                context, citations = run_compression_step(
                    f"compression_contract_retry_{contract_retries}",
                    chunks,
                )
                answer, _ = await run_generation_step(
                    f"generation_contract_retry_{contract_retries}",
                    context,
                    citations,
                    conflict_instruction,
                    allow_flare=False,
                )
                answer, _ = run_firewall_step(
                    f"verification_contract_retry_{contract_retries}",
                    answer,
                    chunks,
                )
                # Loop continues for another contract check
                if contract_retries >= max_evidence_contract_retries:
                    break

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

        # Log comprehensive answer quality metrics (SOTA enhancement)
        # We calculate these before retry to capture the initial attempt quality
        ragas_metrics = self._ragas_evaluator.evaluate(query, answer, chunks)
        invocation_metrics = self._invocation_evaluator.evaluate(
            query=query,
            answer=answer,
            did_retrieve=True,
            chunks_used=len(chunks),
            answer_quality=ragas_metrics.overall_score,
        )

        self._decision_logger.log_answer_quality(
            query=query,
            answer_length=len(answer),
            chunks_used=len(chunks),
            faithfulness=ragas_metrics.faithfulness,
            answer_relevance=ragas_metrics.answer_relevance,
            context_precision=ragas_metrics.context_precision,
            context_recall=ragas_metrics.context_recall,
            overall_score=ragas_metrics.overall_score,
            reflection_quality=reflection_result.quality.value if reflection_result else "unknown",
            should_retry=reflection_result.should_retry if reflection_result else False,
        )
        
        # Update final training outcome
        self._decision_logger.update_outcome(
            query=query,
            outcome_quality=ragas_metrics.overall_score,
        )

        # Feed telemetry back into learned router for continual training signal
        if learned_route and hasattr(self._learned_router, "record_outcome"):
            try:
                current_duration_ms = sum(s.duration_ms for s in pipeline_steps)
                success = ragas_metrics.overall_score >= 0.65
                self._learned_router.record_outcome(
                    query=query,
                    features=learned_route.features,
                    decision=learned_route.decision,
                    success=success,
                    answer_quality=ragas_metrics.overall_score,
                    latency_ms=current_duration_ms,
                    chunks_used=len(chunks),
                )
            except Exception:
                # Training signal is best-effort and should not block responses
                pass

        if reflection_result.should_retry and provider is not None:
            retry_top_k = max(6, int(plan.steps[0].dense_k * 1.5)) if plan.steps else 6
            if on_stage:
                on_stage("retrieval_retry")
            retry_retrieval_start_time = datetime.utcnow()
            retry_retrieval_start = time.perf_counter()
            retry_evidence = self._gatherer.gather(
                query,
                top_k=retry_top_k,
                document_ids=document_ids,
                routing_params=routing_params,
            )
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
            context, citations = run_compression_step("compression_retry", chunks)

            if on_stage:
                on_stage("generation_retry")
            answer, _ = await run_generation_step(
                "generation_retry",
                context,
                citations,
                conflict_instruction,
                allow_flare=False,
                extra_details={"retry": True},
            )

        # Calculate final metrics
        total_tokens = len(context.split()) if context else 0
        coverage = 0.0
        if plan.steps:
            coverage = len(chunks) / plan.steps[0].dense_k

        total_duration_ms = sum(s.duration_ms for s in pipeline_steps)

        trace = self._telemetry.record(
            prompt=query,
            answer=answer,
            metrics={
                "chunks": len(chunks),
                "coverage": coverage,
                "coverage_ratio": final_coverage_ratio,
                "coverage_target": target_coverage,
                "tokens": total_tokens,
                "duration_ms": total_duration_ms,
                "cache_hit": False,
                "answer_quality": reflection_result.quality.value if reflection_result else "unknown",
                "answer_confidence": reflection_result.confidence if reflection_result else 0.0,
                "ragas": ragas_metrics.to_dict(),
                "invocation": invocation_metrics.to_dict(),
                "colbert_enabled": colbert_enabled,
                "precision_stats": precision_stats,
                "rerank_enabled": rerank_enabled,
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
                "coverage_ratio": final_coverage_ratio,
                "coverage_target": target_coverage,
                "tokens": total_tokens,
                "duration_ms": total_duration_ms,
                "query_type": str(query_type),
                "cache_hit": False,
                "answer_quality": reflection_result.quality.value if reflection_result else "unknown",
                "answer_confidence": reflection_result.confidence if reflection_result else 0.0,
                "ragas": ragas_metrics.to_dict(),
                "invocation": invocation_metrics.to_dict(),
                "colbert_enabled": colbert_enabled,
                "precision_stats": precision_stats,
                "rerank_enabled": rerank_enabled,
            },
            "steps": steps_out,
        }
        cacheable = {**result, "steps": [s for s in steps_out if s["name"] != "cache"]}
        cache_manager.queries.set(query, cacheable, cache_hash)
        return result
    
    async def _generate_direct_answer(
        self,
        query: str,
        pipeline_start: datetime,
        pipeline_steps: list[PipelineStep],
        cache_hash: str,
        on_step: Callable[[PipelineStep], None] | None,
        on_token: Callable[[str], None] | None,
        on_stage: Callable[[str], None] | None,
        record_step: Callable[[PipelineStep], None],
    ) -> dict:
        """Generate answer directly without retrieval (for simple queries LLM can handle)."""
        if on_stage:
            on_stage("generation")
        
        gen_start_time = datetime.utcnow()
        gen_start = time.perf_counter()
        provider = self._provider
        
        gen_details: dict[str, Any] = {
            "provider": None,
            "model": None,
            "mode": "direct_no_retrieval",
        }
        
        if provider is None:
            answer = "No provider configured. Please select an LLM provider."
            gen_details["provider"] = "none"
            gen_details["status"] = "error"
        else:
            system_prompt = """You are JR AutoRAG assistant. 
The user's query doesn't require document retrieval - answer it directly from your knowledge.
Be helpful, accurate, and concise."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            
            gen_details["provider"] = getattr(provider, "base_url", "unknown")
            gen_details["model"] = getattr(provider, "default_model", "unknown")
            
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
        
        total_duration_ms = sum(s.duration_ms for s in pipeline_steps)
        
        trace = self._telemetry.record(
            prompt=query,
            answer=answer,
            metrics={
                "chunks": 0,
                "coverage": 0.0,
                "tokens": 0,
                "duration_ms": total_duration_ms,
                "cache_hit": False,
                "mode": "direct_no_retrieval",
            },
            steps=pipeline_steps,
            started_at=pipeline_start,
        )
        
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
            "chunks": [],
            "sources": [],
            "trace_id": trace.id,
            "metrics": {
                "chunks": 0,
                "coverage": 0.0,
                "tokens": 0,
                "duration_ms": total_duration_ms,
                "cache_hit": False,
                "mode": "direct_no_retrieval",
            },
            "steps": steps_out,
        }
        
        cache_manager = get_cache_manager()
        cacheable = {**result, "steps": [s for s in steps_out if s["name"] != "cache"]}
        cache_manager.queries.set(query, cacheable, cache_hash)
        return result
    
    def _build_clarification_response(
        self,
        query: str,
        clarification_question: str | None,
        pipeline_start: datetime,
        pipeline_steps: list[PipelineStep],
    ) -> dict:
        """Build response asking user for clarification."""
        total_duration_ms = sum(s.duration_ms for s in pipeline_steps)
        
        clarification = clarification_question or "Could you provide more context for your question?"
        answer = f"I need a bit more information to answer your question effectively.\n\n**Clarification needed:** {clarification}"
        
        trace = self._telemetry.record(
            prompt=query,
            answer=answer,
            metrics={
                "chunks": 0,
                "coverage": 0.0,
                "tokens": 0,
                "duration_ms": total_duration_ms,
                "cache_hit": False,
                "mode": "clarification_needed",
            },
            steps=pipeline_steps,
            started_at=pipeline_start,
        )
        
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
        
        return {
            "answer": answer,
            "chunks": [],
            "sources": [],
            "trace_id": trace.id,
            "metrics": {
                "chunks": 0,
                "coverage": 0.0,
                "tokens": 0,
                "duration_ms": total_duration_ms,
                "cache_hit": False,
                "mode": "clarification_needed",
                "clarification_question": clarification,
            },
            "steps": steps_out,
            "needs_clarification": True,
        }
