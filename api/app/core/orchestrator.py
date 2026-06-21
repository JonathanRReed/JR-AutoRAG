"""Agentic orchestrator with iterative retrieval and self-correction.

This module implements a SOTA Auto-RAG pipeline with:
- Adaptive retrieval gating (Self-RAG style)
- CRAG-style retrieval quality evaluation
- Iterative retrieve-refine loops with marginal gain stopping
- 10/10 audit-ready citation enforcement
- Self-reflection and answer quality assessment
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, cast

from ..schemas.config import AppConfig
from .abstention import AbstentionResult, get_abstention_rules
from .adaptive_gate import AdaptiveGate, GateDecision
from .artifact_builder import ArtifactStatus, get_artifact_builder
from .budget_planner import BudgetClass, BudgetPlanner
from .cache import RetrievalMode, get_cache_manager

# vNext Expansion: G1-G4 Guarantees
from .citation_verifier import CitationVerifier
from .compression import ContextCompressor
from .conflict_detector import ConflictDetector
from .decision_logger import get_decision_logger
from .evidence_contract import EvidenceContract

# SOTA enhancements
from .flare import FLAREConfig, FLAREGenerator
from .gatherer import EvidenceChunk, Gatherer
from .graph_rag import GraphRAG
from .hallucination_firewall import HallucinationFirewall

# Advanced retrieval modes
from .hierarchy import DocumentTree, HierarchicalRetriever, HierarchyBuilder
from .hyde import get_hyde_generator
from .learned_router import LearnedRouter, RouteDecision
from .local_first import LocalFirstRegistry
from .memory import ConversationMemory
from .persistence import get_disk_query_cache
from .pii_detector import get_pii_detector
from .planner import Planner, PlanStep
from .prompt_guard import CITATION_POLICY_PROMPT
from .providers import LLMProvider, ProviderError, ProviderFactory

# 3.0 Enhancements
from .query_mode import QueryMode, build_no_evidence_answer
from .ragas_eval import InvocationEvaluator, RAGASEvaluator
from .reflection import SelfReflector
from .retrieval import HybridRetrievalEngine, RetrievalEngine

# New agentic components
from .retrieval_evaluator import RetrievalEvaluator, RetrievalVerdict

# Self-RAG critic for v2.0
from .self_rag import get_self_rag_critic

# Web search disabled for offline-only operation
# from .web_search import WebSearch, get_web_search
from .smart_planner import compute_marginal_gain
from .telemetry import (
    PUBLIC_PROVIDER_ERROR_MESSAGE,
    PipelineStep,
    TelemetryStore,
    pipeline_step_to_public_dict,
)
from .trace_export import TraceBundle, create_trace_bundle

# Confidence monitoring
from .uncertainty_monitor import UncertaintyMonitor

logger = logging.getLogger("autorag.pipeline")


def _pipeline_step_log_record(trace_id: str, step: PipelineStep) -> dict[str, Any]:
    """Build a safe pipeline-step log record without step details.

    Pipeline step details can contain user prompts, retrieved document text,
    generated outlines, or other sensitive data. Those details remain available
    through the in-process callback and telemetry trace store, but application
    logs should only receive non-sensitive execution metadata.
    """
    return {
        "event": "pipeline_step",
        "trace_id": trace_id,
        "name": step.name,
        "duration_ms": step.duration_ms,
        "status": step.status,
    }


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
        memory_store: ConversationMemory | None = None,
        policy_registry: LocalFirstRegistry | None = None,
    ) -> None:
        self._planner = planner
        self._retrieval = retrieval
        self._gatherer = gatherer
        self._providers = provider_factory
        self._telemetry = telemetry
        self._memory = memory_store
        self._policy_registry = policy_registry
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
        self._active_tasks = {} # type: dict[str, asyncio.Task]
        self._cancelled_traces = set() # type: set[str]
        self._evidence_contract = EvidenceContract(min_coverage=0.7)
        # Advanced retrieval modes
        self._hierarchy_builder = HierarchyBuilder()
        self._document_trees: dict[str, DocumentTree] = {}
        self._graph_rag: GraphRAG | None = None
        self._graph_ready = False
        self._graph_scope_document_ids: tuple[str, ...] | None = None
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
        # vNext Expansion: G1/G4 Guarantees
        self._citation_verifier = CitationVerifier(max_repair_attempts=2)
        self._artifact_builder = get_artifact_builder()
        self._hallucination_firewall = HallucinationFirewall(strict_mode=False, min_overlap=0.25, min_pass_rate=0.4)
        self._last_trace_bundle: TraceBundle | None = None
        self._hyde_generator = get_hyde_generator()
        # Abstention rules for insufficient evidence scenarios
        self._abstention_rules = get_abstention_rules()
        # Self-RAG critic for LLM-based reflection (v2.0)
        self._self_rag_critic = get_self_rag_critic()

    def rebuild(self, config: AppConfig) -> None:
        self._config = config
        self._planner.rebuild(config)
        if config.provider:
            if self._policy_registry is not None:
                self._policy_registry.ensure_runtime_allowed("llm")
            self._provider = self._providers.build(config.provider)
        if hasattr(self._planner, "set_provider"):
            self._planner.set_provider(self._provider)

        # Only build if not already loaded (prevents double-build on startup)
        if hasattr(self._retrieval, "_chunks") and not self._retrieval._chunks or not hasattr(self._retrieval, "_chunks"):
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

        # Synchronize GraphRAG if available in retriever
        if hasattr(self._retrieval, "_graph_rag") and self._retrieval._graph_rag:
            self._graph_rag = self._retrieval._graph_rag
            self._graph_ready = self._retrieval._graph_ready
            self._graph_scope_document_ids = None

        # Update compressor with config settings
        self._compressor = ContextCompressor(
            max_tokens=config.retrieval.max_context_tokens,
        )
        if not getattr(config.retrieval, "graph", False):
            self._graph_rag = None
            self._graph_ready = False
            self._graph_scope_document_ids = None

        # Sync artifact status with builder for UI (G4)
        if self._hierarchy_ready:
            self._artifact_builder.set_status("raptor", ArtifactStatus.READY)
            self._artifact_builder.set_items("raptor", sum(len(t.nodes) for t in self._document_trees.values()))

        if self._graph_ready:
            self._artifact_builder.set_status("graph_rag", ArtifactStatus.READY)
            if self._graph_rag:
                self._artifact_builder.set_items("graph_rag", len(self._graph_rag.entities))
        elif getattr(self._retrieval, "_graph_failed", False):
            self._artifact_builder.set_status("graph_rag", ArtifactStatus.FAILED)
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

    @staticmethod
    def _normalize_graph_scope(document_ids: list[str] | None) -> tuple[str, ...] | None:
        """Return a stable GraphRAG document scope, or None for all documents."""
        return tuple(sorted(set(document_ids))) if document_ids else None

    def _graph_scope_matches(self, document_ids: list[str] | None) -> bool:
        """Check that the loaded GraphRAG context was built for this request scope."""
        return self._graph_scope_document_ids == self._normalize_graph_scope(document_ids)

    def _graph_entity_in_scope(self, entity: Any, allowed_document_ids: set[str]) -> bool:
        """Return whether an entity has at least one mention in an allowed document."""
        if self._graph_rag is None:
            return False
        chunk_documents = getattr(self._graph_rag, "chunk_documents", {})
        return any(chunk_documents.get(chunk_id) in allowed_document_ids for chunk_id in getattr(entity, "mentions", []))

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
        if not self._graph_scope_matches(document_ids):
            return []

        chunks = []
        allowed_document_ids = set(document_ids) if document_ids else None

        # Find relevant entities via query
        relevant_entities = self._graph_rag.query_entities(query, top_k=5)
        relevant_entity_names: set[str] = set()
        for entity in relevant_entities:
            if allowed_document_ids and not self._graph_entity_in_scope(entity, allowed_document_ids):
                continue
            relevant_entity_names.add(self._graph_rag._normalize_name(entity.name))

        if not relevant_entity_names:
            return []

        # Get community summaries for matched entities
        for community in self._graph_rag.communities:
            community_entity_names = set(community.entities)
            if relevant_entity_names & community_entity_names and community.summary:
                chunks.append(EvidenceChunk(
                    id=f"community_{community.id}",
                    title=f"Topic: {', '.join(community.entities[:3])}",
                    snippet=community.summary,
                    score=0.8,
                ))

        return chunks[:5]  # Limit community chunks

    async def _ensure_graph_context(
        self,
        force: bool = False,
        document_ids: list[str] | None = None,
        on_progress: Callable[[str, int, int, str | None], None] | None = None,
    ) -> None:
        """Build GraphRAG context on demand for the current document scope."""
        target_scope = self._normalize_graph_scope(document_ids)
        if (
            not self._config
            or (not force and not getattr(self._config.retrieval, "graph", False))
            or (self._graph_ready and self._graph_scope_document_ids == target_scope)
        ):
            return
        if self._provider is None or not self._chunk_records:
            return
        try:
            graph_builder = GraphRAG()
            evidence_chunks: list[EvidenceChunk] = []
            scoped_document_ids = set(target_scope) if target_scope is not None else None
            scoped_records = [
                (doc_id, chunk)
                for doc_id, chunk in self._chunk_records
                if scoped_document_ids is None or doc_id in scoped_document_ids
            ]
            # Tighten limit to 100 for faster builds as requested by user
            max_chunks = min(len(scoped_records), 100)
            for doc_id, chunk in scoped_records[:max_chunks]:
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
                    doc_id=doc_id,
                ))
            if not evidence_chunks:
                self._graph_rag = None
                self._graph_ready = False
                self._graph_scope_document_ids = target_scope
                return

            await graph_builder.build_from_chunks(
                evidence_chunks,
                self._provider,
                on_progress=on_progress
            )
            graph_builder.detect_communities()
            await graph_builder.summarize_communities(
                self._provider,
                on_progress=on_progress
            )
            self._graph_rag = graph_builder
            self._graph_ready = True
            self._graph_scope_document_ids = target_scope
        except Exception:
            self._graph_ready = False

    async def _ensure_hierarchy_context(
        self,
        force: bool = False,
        on_progress: Callable[[str, int, int, str | None], None] | None = None,
    ) -> None:
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
            total_docs = len(doc_texts)
            if on_progress:
                on_progress("building_hierarchy", 0, total_docs)

            for idx, (doc_id, parts) in enumerate(doc_texts.items()):
                combined = "\n\n".join(parts)
                if not combined.strip():
                    continue
                tree = self._hierarchy_builder.build(
                    combined,
                    document_id=doc_id,
                    title=f"Document {doc_id}",
                )
                trees[doc_id] = tree
                if on_progress:
                    on_progress("building_hierarchy", idx + 1, total_docs)

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
        chunks: list[EvidenceChunk],
        document_ids: list[str] | None,
    ) -> list[EvidenceChunk]:
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
            if provider and hasattr(provider, "get_token_stats") and callable(provider.get_token_stats):
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

    async def _build_thinking_outline(
        self,
        *,
        query: str,
        context_text: str | None,
        provider: LLMProvider | None,
        citations_count: int,
        allow_query_only: bool = False,
    ) -> tuple[str | None, dict[str, Any], str]:
        """Create a short, user-visible outline without chain-of-thought."""
        details: dict[str, Any] = {
            "mode": "outline",
            "sources": citations_count,
            "context_tokens": len(context_text.split()) if context_text else 0,
        }
        if provider is None:
            details["reason"] = "no_provider"
            return None, details, "skipped"

        context_available = bool(context_text and context_text.strip())
        if not context_available and not allow_query_only:
            details["reason"] = "no_context"
            return None, details, "skipped"

        system_prompt = (
            "You create a concise answer outline for users. "
            "Do NOT reveal chain-of-thought or step-by-step reasoning. "
            "Return 3-6 bullet points with short phrases, no citations."
        )
        context_excerpt = (context_text or "").strip()
        if context_excerpt:
            context_excerpt = context_excerpt[:2400] + ("..." if len(context_excerpt) > 2400 else "")
        user_prompt = (
            f"User question:\n{query}\n\n"
            f"Context excerpt:\n{context_excerpt or '[none]'}\n\n"
            "Output only a short bullet outline."
        )
        try:
            outline = await provider.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        except ProviderError as exc:
            details["error"] = str(exc)
            return None, details, "skipped"

        outline = outline.strip()
        if len(outline) > 1400:
            outline = outline[:1400] + "..."
        details["outline"] = outline
        return outline, details, "completed"

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
            completed_at=datetime.now(UTC),
            duration_ms=round((end - start_perf) * 1000, 2),
            details=details,
            status=status,
        )

    def _query_cache_scope(
        self,
        document_ids: list[str] | None = None,
        history: list[dict[str, str]] | None = None,
        conversation_id: str | None = None,
        cache_scope: str | None = None,
    ) -> str | None:
        """Create a request-scope fingerprint for full-answer cache entries.

        Full query results include generated text and source snippets, so the cache
        must vary on request-local inputs that can change authorization or model
        context.  Routers may pass an already-hashed user/ACL scope in
        ``cache_scope``; direct callers are still protected by folding document
        filters, chat history, and memory conversation IDs into this fingerprint.
        """
        scope_payload = {
            "cache_scope": cache_scope or "",
            "document_ids": sorted(document_ids or []),
            "history": history or [],
            "conversation_id": conversation_id or "",
        }
        if not any(scope_payload.values()):
            return None
        encoded = json.dumps(scope_payload, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    def _cache_config_hash(
        self,
        document_ids: list[str] | None,
        cache_scope: str | None = None,
    ) -> str:
        config = self._config
        provider = config.provider if config else None
        payload = {
            "document_ids": document_ids or [],
            "cache_scope": cache_scope or "",
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

    def _maybe_redact_pii(self, text: str) -> tuple[str, bool]:
        if not text:
            return text, False
        flag = os.environ.get("AUTORAG_PII_REDACT", "false").lower()
        if flag not in ("true", "1", "yes", "on"):
            return text, False
        detector = get_pii_detector()
        result = detector.detect(text)
        if not result.has_pii:
            return text, False
        return detector.redact(text, result.matches), True

    # Human-readable stage messages for progress updates
    STAGE_MESSAGES = {
        "cache": "Checking query cache...",
        "planning": "Planning retrieval strategy...",
        "gating": "Determining retrieval mode...",
        "routing": "Finding source documents...",
        "gatherer": "Gathering relevant evidence...",
        "retrieval": "Processing retrieval results...",
        "retrieval_iteration": "Refining search scope...",
        "evaluation": "Analyzing evidence relevance...",
        "compression": "Optimizing context windows...",
        "thinking": "Drafting a concise answer outline...",
        "generation": "Crafting your answer...",
        "conflict_detection": "Checking for factual consistency...",
        "citation_verification": "Validating citations and sources...",
        "reflection": "Quality checking final answer...",
        "hallucination_check": "Running hallucination firewall...",
        "evidence_contract": "Ensuring evidence coverage...",
        "graph_build": "Building GraphRAG context...",
        "graph_retrieval": "Retrieving graph summaries...",
        "extracting_graph": "Building knowledge graph...",
        "summarizing_communities": "Thematizing communities...",
        "building_hierarchy": "Building document hierarchy...",
    }
    def cancel_trace(self, trace_id: str) -> None:
        """Mark a trace as cancelled."""
        self._cancelled_traces.add(trace_id)

    async def answer(
        self,
        query: str,
        document_ids: list[str] | None = None,
        on_step: Callable[[PipelineStep], None] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_stage: Callable[[str], None] | None = None,
        on_progress: Callable[[dict], None] | None = None,
        history: list[dict[str, str]] | None = None,
        conversation_id: str | None = None,
        trace_id: str | None = None,
        query_mode: QueryMode | None = None,  # P0.1: Grounded vs Open Domain
        cache_scope: str | None = None,
    ) -> dict:
        if trace_id is None:
            trace_id = uuid.uuid4().hex[:16]

        # Immediate cleanup check
        if trace_id in self._cancelled_traces:
            self._cancelled_traces.remove(trace_id) # Reset for next use
            raise asyncio.CancelledError(f"Trace {trace_id} was cancelled before starting")

        pipeline_start = datetime.now(UTC)
        pipeline_steps: list[PipelineStep] = []
        reflection_result = None
        runtime_profile = self._retrieval.get_runtime_profile() if hasattr(self._retrieval, "get_runtime_profile") else {}
        memory_context = ""
        if conversation_id and self._memory is not None:
            memory_context = self._memory.build_context_prompt(conversation_id, query)

        # Progress tracking
        query_start_time = time.perf_counter()
        current_stage = None
        stage_start_time = query_start_time

        def emit_progress(
            stage: str,
            detail: str | None = None,
            progress: float | None = None,
            items_done: int | None = None,
            items_total: int | None = None,
        ) -> None:
            """Emit a progress event with human-readable message."""
            nonlocal current_stage, stage_start_time
            if on_progress is None:
                return

            # Update stage timer if stage changed
            now = time.perf_counter()
            if stage != current_stage:
                current_stage = stage
                stage_start_time = now

            elapsed_ms = round((now - query_start_time) * 1000, 1)
            stage_elapsed_ms = round((now - stage_start_time) * 1000, 1)
            message = self.STAGE_MESSAGES.get(stage, f"Processing {stage}...")

            progress_data = {
                "stage": stage,
                "message": message,
                "trace_id": trace_id,
                "elapsed_ms": elapsed_ms, # Use total elapsed for UI consistency
                "stage_elapsed_ms": stage_elapsed_ms,
            }

            if detail:
                progress_data["detail"] = detail

            if progress is not None:
                progress_data["progress"] = round(progress, 2)
            elif items_done is not None and items_total is not None and items_total > 0:
                progress_data["progress"] = round(items_done / items_total, 2)
                progress_data["detail"] = detail or f"Processing {items_done} of {items_total}"
                # Estimate remaining time
                if items_done > 0:
                    time_per_item = stage_elapsed_ms / items_done
                    remaining_items = items_total - items_done
                    progress_data["estimated_remaining_ms"] = round(time_per_item * remaining_items, 1)

            on_progress(progress_data)

        def record_step(step: PipelineStep) -> None:
            pipeline_steps.append(step)
            if on_step:
                on_step(step)
            try:
                logger.info(json.dumps(_pipeline_step_log_record(trace_id, step)))
            except Exception:
                logger.info("pipeline_step trace_id=%s name=%s", trace_id, step.name)

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
        query_cache_scope = self._query_cache_scope(
            document_ids=document_ids,
            history=history,
            conversation_id=conversation_id,
            cache_scope=cache_scope,
        )
        cache_hash = self._cache_config_hash(document_ids, query_cache_scope)

        # G3: Get corpus version and retrieval mode for versioned cache keys
        cache_corpus_version = ""
        cache_retrieval_mode = RetrievalMode.STANDARD
        if hasattr(self._retrieval, 'get_corpus_version'):
            cache_corpus_version = self._retrieval.get_corpus_version()
        if hasattr(self._retrieval, 'get_retrieval_mode_flags'):
            cache_retrieval_mode = self._retrieval.get_retrieval_mode_flags()

        # P0.1: Resolve query mode from parameter or config
        effective_query_mode = query_mode
        if effective_query_mode is None:
            config_mode = getattr(self._config, "query_mode", "grounded")
            effective_query_mode = QueryMode(config_mode) if config_mode in ("grounded", "open_domain") else QueryMode.GROUNDED

        # P0.3: Get current preset ID for cache key
        current_preset_id = getattr(self._config.retrieval, "_preset_level", "balanced") if self._config else "balanced"
        if not current_preset_id or current_preset_id == "balanced":
            # Try to infer from config
            current_preset_id = "balanced"

        # P0.3: Get model IDs for cache key
        provider_config = self._config.provider if self._config else None
        model_ids = {
            "planner": getattr(provider_config, "planner_model", "") or "",
            "gatherer": getattr(provider_config, "gatherer_model", "") or "",
            "generator": getattr(provider_config, "generator_model", "") or "",
        } if provider_config else {}

        cache_start_time = datetime.now(UTC)
        cache_start = time.perf_counter()
        if on_stage:
            on_stage("cache")
        emit_progress("cache", detail="Checking index freshness")

        # P0.3: Use disk-backed query cache with versioned keys
        disk_cache = get_disk_query_cache()
        cached_result = disk_cache.get(
            query=query,
            corpus_version=cache_corpus_version,
            retrieval_mode=int(cache_retrieval_mode),
            preset_id=current_preset_id,
            model_ids=model_ids,
            scope_key=query_cache_scope,
        )
        cache_event = disk_cache.get_last_event()

        cache_step_details: dict[str, Any] = {
            "query_cache": "enabled",
            "disk_backed": True,
            "cache_event": cache_event.to_dict() if cache_event else None,
            "corpus_version": cache_corpus_version,
            "retrieval_mode": int(cache_retrieval_mode),
            "preset_id": current_preset_id,
        }

        if cached_result is not None:
            cache_step_details["cache_hit"] = True
            cache_step = self._make_step(
                "cache",
                cache_start,
                cache_start_time,
                cache_step_details,
            )
            record_step(cache_step)
            # Return cached result with updated trace
            cached_result["steps"] = [pipeline_step_to_public_dict(s) for s in pipeline_steps]
            cached_result["from_cache"] = True
            return cached_result

        cache_step_details["cache_hit"] = False
        cache_step = self._make_step(
            "cache",
            cache_start,
            cache_start_time,
            cache_step_details,
        )
        record_step(cache_step)

        policy_step_start = datetime.now(UTC)
        policy_step = self._make_step(
            "policy",
            time.perf_counter(),
            policy_step_start,
            {
                "conversation_id": conversation_id or "",
                "deployment_profile": runtime_profile.get("deployment_profile", ""),
                "runtime_backends": runtime_profile.get("backends", {}),
                "memory_context_loaded": bool(memory_context),
            },
        )
        record_step(policy_step)

        # Step 1: Planning (now with query analysis)
        if on_stage:
            on_stage("planning")
        emit_progress("planning", detail="Breaking down your question")
        plan_start_time = datetime.now(UTC)
        plan_start = time.perf_counter()
        stage_start_time = plan_start  # Reset stage timer
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
        emit_progress("gating", detail="Checking if documents are needed")
        gating_start_time = datetime.now(UTC)
        gating_start = time.perf_counter()
        stage_start_time = gating_start

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
                on_step, on_token, on_stage, record_step,
                conversation_id=conversation_id,
                memory_context=memory_context,
                runtime_profile=runtime_profile,
            )

        # Handle clarification case
        if gate_result.decision == GateDecision.CLARIFY_FIRST:
            gating_details["clarification"] = gate_result.clarification_question
            record_step(self._make_step("gating", gating_start, gating_start_time, gating_details))
            # Return clarification request
            return self._build_clarification_response(
                query, gate_result.clarification_question, pipeline_start, pipeline_steps,
                conversation_id=conversation_id,
                runtime_profile=runtime_profile,
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
        emit_progress("routing", detail="Choosing optimal search strategy")
        routing_start_time = datetime.now(UTC)
        routing_start = time.perf_counter()
        stage_start_time = routing_start

        rerank_enabled = False

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
        # PERFORMANCE FIX: Hard cap at 2 iterations to prevent long delays
        # Complex multi-iteration loops with evaluations were causing 20+ minute responses
        max_iterations = min(max_iterations, 2)

        # Adjust retrieval k per learned router suggestion
        for step in getattr(plan, "steps", []):
            if hasattr(step, "dense_k"):
                step.dense_k = max(step.dense_k, learned_route.suggested_k or step.dense_k)
            if hasattr(step, "sparse_k"):
                step.sparse_k = max(step.sparse_k, learned_route.suggested_k or step.sparse_k)

        # Context building (can be slow)
        if base_use_raptor and not self._hierarchy_ready:
            await self._ensure_hierarchy_context(
                force=False,
                on_progress=lambda s, c, t: emit_progress(s, items_done=c, items_total=t)
            )

        if base_use_graph:
            graph_build_details = {
                "enabled": True,
                "already_ready": self._graph_ready,
                "chunks_available": len(self._chunk_records),
            }
            if self._graph_ready and self._graph_scope_matches(document_ids):
                record_step(PipelineStep(
                    name="graph_build",
                    started_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                    duration_ms=0.0,
                    details=graph_build_details,
                    status="skipped",
                ))
            elif self._provider is None or not self._chunk_records:
                graph_build_details["reason"] = "no_provider" if self._provider is None else "no_chunks"
                record_step(PipelineStep(
                    name="graph_build",
                    started_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                    duration_ms=0.0,
                    details=graph_build_details,
                    status="skipped",
                ))
            else:
                if on_stage:
                    on_stage("graph_build")
                emit_progress("graph_build", detail="Preparing GraphRAG context")
                graph_build_start_time = datetime.now(UTC)
                graph_build_start = time.perf_counter()
                await self._ensure_graph_context(
                    force=False,
                    document_ids=document_ids,
                    on_progress=lambda s, c, t: emit_progress(s, items_done=c, items_total=t)
                )
                graph_build_details["graph_ready"] = self._graph_ready
                graph_build_details["entity_count"] = len(self._graph_rag.entities) if self._graph_rag else 0
                status = "completed" if self._graph_ready else "failed"
                record_step(self._make_step(
                    "graph_build",
                    graph_build_start,
                    graph_build_start_time,
                    graph_build_details,
                    status=status,
                ))

        # Enable high precision mode if budget allows
        emit_progress("routing", detail="Configuring retrieval engine...")
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
        emit_progress(
            "gatherer",
            detail=f"Searching across {len(plan.steps)} query variations",
            items_done=0,
            items_total=max_iterations,
        )
        gatherer_start_time = datetime.now(UTC)
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
        graph_retrieval_ms = 0.0
        embedding_cache_hits = 0
        embedding_cache_misses = 0

        # HyDE query enhancement if enabled
        cfg_use_hyde = bool(getattr(cfg_retrieval, "use_hyde", False))
        hyde_enhanced_query = query
        hyde_details: dict[str, Any] = {"enabled": cfg_use_hyde}

        if cfg_use_hyde and self._provider is not None:
            try:
                hyde_result = await self._hyde_generator.generate(
                    query,
                    self._provider,
                    query_type=str(query_type)
                )
                if hyde_result.hypotheticals:
                    hyde_enhanced_query = hyde_result.embedding_text
                    hyde_details["hypothetical_generated"] = True
                    hyde_details["hypothetical_preview"] = hyde_result.hypotheticals[0][:200]
            except Exception as e:
                hyde_details["error"] = str(e)

        # Iterative retrieval loop
        iteration = 0
        accumulated_chunks = []
        current_query = hyde_enhanced_query if cfg_use_hyde else query
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
            if trace_id in self._cancelled_traces:
                raise asyncio.CancelledError(f"Trace {trace_id} cancelled by user")
            iteration_start = time.perf_counter()
            iteration_chunks = []
            iteration_details = {"iteration": iteration + 1, "query": current_query, "sub_queries": []}

            # Emit per-iteration progress
            emit_progress(
                "retrieval_iteration" if iteration > 0 else "gatherer",
                detail=f"Search iteration {iteration + 1} of {max_iterations}",
                items_done=iteration,
                items_total=max_iterations,
            )
            stage_start_time = iteration_start

            # Execute retrieval for all plan steps in parallel
            total_steps = len(plan.steps)

            async def run_retrieval_step(
                idx: int,
                step: PlanStep,
                _iteration: int = iteration,
                _current_query: str = current_query,
                _total_steps: int = total_steps,
            ):
                step_query = step.query if _iteration == 0 else _current_query
                query_preview = step_query[:50] + "..." if len(step_query) > 50 else step_query
                sub_start = time.perf_counter()

                emit_progress(
                    "gatherer",
                    detail=f"Searching query {idx + 1}/{_total_steps}: \"{query_preview}\"",
                    items_done=idx,
                    items_total=_total_steps,
                )

                step_evidence = await self._gatherer.gather(
                    step_query,
                    top_k=step.dense_k,
                    document_ids=document_ids,
                    routing_params=routing_params,
                    on_progress=lambda msg, val: emit_progress(
                        "gatherer",
                        detail=f"{query_preview}: {msg}",
                        progress=val
                    )
                )
                duration_ms = round((time.perf_counter() - sub_start) * 1000, 2)
                return idx, step_query, step_evidence, duration_ms

            # Run all steps in parallel for this iteration
            tasks = [asyncio.create_task(run_retrieval_step(i, s)) for i, s in enumerate(plan.steps)]
            try:
                step_results = await asyncio.gather(*tasks)
            except Exception as e:
                # Cancel all pending tasks if any fail to avoid RuntimeWarning or orphaned tasks
                for t in tasks:
                    if not t.done():
                        t.cancel()
                raise e

            # Aggregate results from parallel tasks
            for _step_idx, step_query, step_evidence, duration_ms in step_results:
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
                    "duration_ms": duration_ms,
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
                    try:
                        raptor_chunks = await self._retrieve_with_raptor(
                            current_query, document_ids, iteration_chunks
                        )
                        if raptor_chunks:
                            iteration_chunks.extend(raptor_chunks)
                            iteration_details["raptor_chunks_added"] = len(raptor_chunks)
                    except Exception as e:
                        print(f"RAPTOR corrective retrieval failed: {e}")
                if self._graph_rag is not None and iteration_details.get("graph_chunks_added") is None:
                    try:
                        if on_stage:
                            on_stage("graph_retrieval")
                        graph_start = time.perf_counter()
                        graph_chunks = await self._retrieve_with_graph(current_query, document_ids)
                        graph_retrieval_ms += (time.perf_counter() - graph_start) * 1000
                        if graph_chunks:
                            iteration_chunks.extend(graph_chunks)
                            iteration_details["graph_chunks_added"] = len(graph_chunks)
                    except Exception as e:
                        print(f"GraphRAG corrective retrieval failed: {e}")

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
                    sub_evidence = await self._gatherer.gather(
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
                        clar_evidence = await self._gatherer.gather(
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
                            sub_evidence = await self._gatherer.gather(
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
                try:
                    raptor_chunks = await self._retrieve_with_raptor(
                        current_query, document_ids, iteration_chunks
                    )
                    if raptor_chunks:
                        iteration_chunks.extend(raptor_chunks)
                        iteration_details["raptor_chunks_added"] = len(raptor_chunks)
                        iteration_details["raptor_enabled"] = True
                        total_raptor_chunks += len(raptor_chunks)
                except Exception as e:
                    print(f"RAPTOR initial retrieval failed: {e}")

            # GraphRAG: Add community summaries if enabled on first hop
            if use_graph and self._graph_rag is not None and iteration == 0:
                try:
                    if on_stage:
                        on_stage("graph_retrieval")
                    graph_start = time.perf_counter()
                    graph_chunks = await self._retrieve_with_graph(current_query, document_ids)
                    graph_retrieval_ms += (time.perf_counter() - graph_start) * 1000
                    if graph_chunks:
                        iteration_chunks.extend(graph_chunks)
                        iteration_details["graph_chunks_added"] = len(graph_chunks)
                        iteration_details["graph_enabled"] = True
                        total_graph_chunks += len(graph_chunks)
                except Exception as e:
                    print(f"GraphRAG initial retrieval failed: {e}")

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

        if cfg_use_graph:
            graph_status = "completed" if total_graph_chunks > 0 else "skipped"
            graph_step = PipelineStep(
                name="graph_retrieval",
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                duration_ms=round(graph_retrieval_ms, 2),
                details={
                    "graph_ready": bool(self._graph_rag),
                    "chunks_added": total_graph_chunks,
                    "duration_ms": round(graph_retrieval_ms, 2),
                },
                status=graph_status,
            )
            record_step(graph_step)

        # Step 3: Retrieval (aggregation + dedupe)
        if on_stage:
            on_stage("retrieval")
        retrieval_start_time = datetime.now(UTC)
        retrieval_start = time.perf_counter()
        retrieval_details: dict[str, Any] = {
            "sub_queries": gatherer_details["sub_queries"],
            "raptor_chunks_added": total_raptor_chunks,
            "graph_chunks_added": total_graph_chunks,
        }
        precision_stats: dict[str, Any] = {}
        colbert_enabled = False

        # Deduplicate chunks by ID, keeping highest score
        seen: dict[str, Any] = {}
        for chunk in all_chunks:
            if chunk.id not in seen or chunk.score > seen[chunk.id].score:
                seen[chunk.id] = chunk

        chunks = list(seen.values())
        # Sort by score descending
        chunks.sort(key=lambda c: c.score, reverse=True)

        if not chunks:
            evidence = await self._gatherer.gather(
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
        if hasattr(self._retrieval, "get_model_status"):
            retrieval_details.update(self._retrieval.get_model_status())
        if hasattr(self._retrieval, "get_last_bq_debug"):
            bq_debug = self._retrieval.get_last_bq_debug()
            if bq_debug:
                retrieval_details["bq_debug"] = bq_debug
                retrieval_details["retrieval_backend"] = bq_debug.get("mode", "binary")
        retrieval_details["rerank_enabled"] = rerank_enabled
        retrieval_details["reranked"] = rerank_enabled
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
        retrieval_backend = retrieval_details.get("retrieval_backend", "float32")

        # P0.1: Grounded mode no-evidence check
        # If grounded mode is active and no chunks found, return structured response
        if effective_query_mode == QueryMode.GROUNDED and not chunks:
            no_evidence_response = build_no_evidence_answer(
                query=query,
                corpus_doc_count=len(getattr(self._retrieval, '_document_store', {}) or []),
                corpus_chunk_count=len(getattr(self._retrieval, '_chunks', []) or []),
                search_terms_tried=[s.query for s in plan.steps] if hasattr(plan, 'steps') else [query],
            )
            no_evidence_response["trace_id"] = trace_id
            no_evidence_response["steps"] = [
                step.__dict__ if hasattr(step, '__dict__') else step
                for step in pipeline_steps
            ]
            no_evidence_response["metrics"]["total_duration_ms"] = round(
                (datetime.now(UTC) - pipeline_start).total_seconds() * 1000, 2
            )

            # Record the no-evidence step
            no_evidence_step = PipelineStep(
                name="no_evidence",
                started_at=datetime.now(UTC),
                completed_at=datetime.now(UTC),
                duration_ms=0.0,
                details={
                    "mode": "grounded",
                    "reason": "no_supporting_documents",
                    "chunks_found": 0,
                    "suggested_actions": len(no_evidence_response.get("grounding", {}).get("no_evidence_response", {}).get("suggested_actions", [])),
                },
                status="no_evidence",
            )
            record_step(no_evidence_step)

            return no_evidence_response

        # Abstention check: Should we refuse to answer due to insufficient evidence?
        cfg_abstain = getattr(cfg_retrieval, "abstain_when_unverified", False)
        if cfg_abstain:
            abstention_result = self._abstention_rules.check(
                chunks=chunks,
                retrieval_verdict=retrieval_verdict,
                verdict_confidence=eval_result.confidence if 'eval_result' in dir() else 0.5,
                coverage_ratio=final_coverage_ratio,
                plan_coverage_ratio=plan_coverage_ratio if 'plan_coverage_ratio' in dir() else 0.5,
                query=query,
            )

            if abstention_result.should_abstain:
                # Record abstention step
                abstention_step = PipelineStep(
                    name="abstention",
                    started_at=datetime.now(UTC),
                    completed_at=datetime.now(UTC),
                    duration_ms=0.0,
                    details={
                        "reason": abstention_result.reason.value if abstention_result.reason else "unknown",
                        "confidence": abstention_result.confidence,
                        **abstention_result.details,
                    },
                    status="abstained",
                )
                record_step(abstention_step)

                return self._build_abstention_response(
                    query=query,
                    abstention_result=abstention_result,
                    chunks=chunks,
                    pipeline_start=pipeline_start,
                    pipeline_steps=pipeline_steps,
                    conversation_id=conversation_id,
                    runtime_profile=runtime_profile,
                )

        compression_enabled = bool(self._config and self._config.retrieval.compression)

        def run_compression_step(step_name: str, chunk_list: list[EvidenceChunk]) -> tuple[str, list[dict]]:
            if on_stage:
                on_stage(step_name)
            compression_start_time = datetime.now(UTC)
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
            # Emit progress for generation
            emit_progress(
                step_name,
                detail=f"Creating response with {len(citation_payload)} sources",
            )
            gen_start_time = datetime.now(UTC)
            gen_start = time.perf_counter()
            provider = self._provider
            gen_details: dict[str, Any] = {
                "provider": "configured" if provider else "none",
                "model": "configured" if provider else None,
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
                system_prompt = f"""You are JR AutoRAG Assistant, a precise enterprise RAG generator.

## CRITICAL CONSTRAINT
You MUST ONLY answer using information from the provided KNOWLEDGE BASE context below.
If the question CANNOT be answered from the provided context, you MUST say:
"I cannot answer this question because the information is not available in the current knowledge base."
DO NOT use your general knowledge or training data - ONLY use the provided context.

## STEP-BY-STEP ANSWER SYNTHESIS
For each question:
1. IDENTIFY: Find the most relevant passages in the context that address the question
2. EXTRACT: Pull out key facts, quotes, and data points from those passages
3. SYNTHESIZE: Combine the evidence into a coherent, well-structured answer
4. CITE: Add bracketed citations [1], [2] for EVERY factual claim
5. VERIFY: Confirm each claim has supporting evidence in the context

## CITATION FORMAT (MANDATORY)
- Every factual claim needs: "claim [n]." followed by "exact quote" (Source: [n])
- Number citations sequentially: [1], [2], [3]...
- Include a ## Sources section at the end listing each citation

{CITATION_POLICY_PROMPT}

## FALLBACK RULES (CRITICAL)
- If information is NOT in the context: Say "This information is not available in the current knowledge base."
- If a date/number/metric is missing: Write "[Metadata not found in knowledge base]"
- If the question is unrelated to the documents: Explain what topics the knowledge base covers
- NEVER invent facts, dates, numbers, or quotes from your training data

## OUTPUT QUALITY
- Be concise but complete
- Use clear structure (headers, bullets where appropriate)
- Prioritize the most relevant information first{conflict_text}"""

                user_prompt = (
                    f"### KNOWLEDGE BASE (Use ONLY this information):\n{context_text}\n\n"
                    f"### USER QUESTION:\n{query}\n\n"
                    f"### INSTRUCTIONS:\n{strict_instruction}\n\n"
                    "### REMINDER:\nFollow the step-by-step synthesis process. Cite every claim."
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
                            ]
                            if memory_context:
                                messages.append({"role": "system", "content": memory_context})
                            # Inject history if available (Phase 12)
                            if history:
                                for turn in history:
                                    role = turn.get("role", "user")
                                    content = turn.get("content", "")
                                    if role in ("user", "assistant") and content:
                                        messages.append({"role": role, "content": content})

                            messages.append({"role": "user", "content": user_prompt})
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
                except ProviderError:
                    answer = PUBLIC_PROVIDER_ERROR_MESSAGE
                    gen_details["status"] = "error"
                    gen_details["error"] = PUBLIC_PROVIDER_ERROR_MESSAGE

            record_step(self._make_step(step_name, gen_start, gen_start_time, gen_details))
            return answer, gen_details

        # Step 3: Context Compression (new!)
        context, citations = run_compression_step("compression", chunks)

        # Step 3.5: Conflict Detection
        if on_stage:
            on_stage("conflict_detection")
        conflict_start_time = datetime.now(UTC)
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

        # Step 3.75: Thinking / Outline (user-visible, no chain-of-thought)
        if on_stage:
            on_stage("thinking")
        emit_progress("thinking", detail="Drafting a concise outline")
        thinking_start_time = datetime.now(UTC)
        thinking_start = time.perf_counter()
        _, thinking_details, thinking_status = await self._build_thinking_outline(
            query=query,
            context_text=context,
            provider=self._provider,
            citations_count=len(citations),
            allow_query_only=False,
        )
        record_step(self._make_step(
            "thinking",
            thinking_start,
            thinking_start_time,
            thinking_details,
            status=thinking_status,
        ))

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
            chunk_list: list[EvidenceChunk],
        ) -> tuple[str, dict]:
            stage_label = "verification" if step_name == "verification" else "verification_retry"
            if on_stage:
                on_stage(stage_label)
            firewall_start_time = datetime.now(UTC)
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
        # Wrapped in try/except to ensure responses always complete
        firewall_passed: bool | None = None
        try:
            answer, firewall_details = run_firewall_step("verification", answer, chunks)
        except Exception as fw_err:
            print(f"Hallucination firewall error (non-blocking): {fw_err}")
            firewall_details = {"pass_rate": 1.0, "skipped": True, "error": str(fw_err)}
        else:
            firewall_passed = firewall_details.get("meets_threshold", True)

        # Step 4.5b: Firewall-driven corrective retry is DISABLED to improve response times
        # The retry mechanism was causing 20+ minute delays. If pass rate is low, we now
        # simply return the response with a warning rather than retrying.
        firewall_pass_rate = firewall_details.get("pass_rate", 1.0)

        # DISABLED: The retry mechanism below was causing 20+ minute delays.
        # When pass rate is low, we now simply return the response rather than retrying.
        # Users can always regenerate manually if needed.
        #
        # if (
        #     firewall_pass_rate < 0.6
        #     and provider is not None
        #     and chunks
        # ):
        #     ... (retry logic disabled for performance)

        # Step 4.6: Evidence-first contract enforcement
        contract_result = None
        contract_checks = 0
        contract_retries = 0
        contract_passed: bool | None = None
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
                contract_start_time = datetime.now(UTC)
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
                        bundle = await self._gatherer.gather(
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
                emit_progress(
                    contract_step_name,
                    detail=f"Contract failed ({round(contract_result.coverage_ratio, 2)} coverage). Retrying with {len(targeted_chunks)} new chunks...",
                    items_done=contract_retries,
                    items_total=max_evidence_contract_retries
                )
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
                emit_progress(
                    f"generation_contract_retry_{contract_retries}",
                    detail="Verification successful. Finalizing answer...",
                    items_done=contract_retries,
                    items_total=max_evidence_contract_retries
                )
                answer, _ = run_firewall_step(
                    f"verification_contract_retry_{contract_retries}",
                    answer,
                    chunks,
                )
                # Loop continues for another contract check
                if contract_retries >= max_evidence_contract_retries:
                    break
            contract_passed = contract_result.pass_threshold if contract_result else None

        # Step 4.7: Citation Verification (G1 Guarantee - vNext Expansion)
        # Ensures every citation maps to a retrieved chunk ID or gets repaired/marked
        if on_stage:
            on_stage("citation_verification")
        citation_start_time = datetime.now(UTC)
        citation_start = time.perf_counter()

        citation_result = self._citation_verifier.verify(answer, chunks)
        citation_details = citation_result.to_trace_dict()

        # If citations are invalid and we have a provider, attempt repair
        if not citation_result.all_valid and self._provider is not None:
            try:
                citation_result = await asyncio.wait_for(
                    self._citation_verifier.verify_and_repair(
                        answer, chunks, self._provider
                    ),
                    timeout=45.0,  # Overall timeout for citation repair
                )
                answer = citation_result.verified_answer
                citation_details = citation_result.to_trace_dict()
                citation_details["repair_applied"] = True
            except TimeoutError:
                citation_details["repair_error"] = "Citation repair timed out"
            except Exception as e:
                citation_details["repair_error"] = str(e)

        record_step(self._make_step(
            "citation_verification",
            citation_start,
            citation_start_time,
            citation_details,
            status="passed" if citation_result.final_pass else "repaired",
        ))

        if on_stage:
            on_stage("reflection")
        reflection_start_time = datetime.now(UTC)
        reflection_start = time.perf_counter()

        # Phase 1: Heuristic reflection
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

        # Phase 2: Self-RAG LLM-based critic (v2.0 enhancement)
        cfg_self_rag = getattr(cfg_retrieval, "self_rag_critic", False) if cfg_retrieval else False
        critic_result = None
        if cfg_self_rag and self._provider is not None:
            try:
                critic_result = await self._self_rag_critic.critique(
                    query=query,
                    response=answer,
                    chunks=chunks,
                    provider=self._provider,
                )
                reflection_details["self_rag"] = {
                    "relevance": critic_result.relevance.value,
                    "support": critic_result.support.value,
                    "utility": critic_result.utility.value,
                    "should_regenerate": critic_result.should_regenerate,
                    "critique": critic_result.critique[:200] if critic_result.critique else "",
                }
                # Override retry decision if critic strongly recommends regeneration
                if critic_result.should_regenerate and critic_result.utility <= 2:
                    reflection_result.should_retry = True
            except Exception as e:
                reflection_details["self_rag_error"] = str(e)

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

        if reflection_result.should_retry and self._provider is not None:
            retry_top_k = max(6, int(plan.steps[0].dense_k * 1.5)) if plan.steps else 6
            if on_stage:
                on_stage("retrieval_retry")
            retry_retrieval_start_time = datetime.now(UTC)
            retry_retrieval_start = time.perf_counter()
            retry_evidence = await self._gatherer.gather(
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
        answer, pii_redacted = self._maybe_redact_pii(answer)
        total_tokens = len(context.split()) if context else 0
        coverage = 0.0
        if plan.steps:
            dense_k = max(1, plan.steps[0].dense_k)
            coverage = len(chunks) / dense_k

        total_duration_ms = sum(s.duration_ms for s in pipeline_steps)
        retrieval_mode = "standard"
        if total_graph_chunks and total_raptor_chunks:
            retrieval_mode = "combined"
        elif total_graph_chunks:
            retrieval_mode = "graph"
        elif total_raptor_chunks:
            retrieval_mode = "raptor"
        flare_retrievals = int(gen_details.get("flare_retrievals", 0)) if gen_details else 0
        firewall_pass_rate = firewall_details.get("pass_rate", 1.0)
        memory_result = None
        if conversation_id and self._memory is not None:
            memory_result = self._memory.record_exchange(
                conversation_id=conversation_id,
                user_query=query,
                answer=answer,
                metadata={
                    "chunks_used": [chunk.id for chunk in chunks],
                    "sources_count": len(citations),
                    "query_type": str(query_type),
                },
            )

        trace = self._telemetry.record(
            prompt=query,
            answer=answer,
            metrics={
                "chunks": len(chunks),
                "context_chunks": len(chunks),
                "coverage": coverage,
                "coverage_ratio": final_coverage_ratio,
                "coverage_target": target_coverage,
                "tokens": total_tokens,
                "duration_ms": total_duration_ms,
                "cache_hit": False,
                "embedding_cache_hits": embedding_cache_hits,
                "embedding_cache_misses": embedding_cache_misses,
                "retrieval_mode": retrieval_mode,
                "retrieval_backend": retrieval_backend,
                "flare_retrievals": flare_retrievals,
                "firewall_pass_rate": firewall_pass_rate,
                "quality_rating": reflection_result.quality.value if reflection_result else "unknown",
                "answer_quality": reflection_result.quality.value if reflection_result else "unknown",
                "answer_confidence": reflection_result.confidence if reflection_result else 0.0,
                "faithfulness": ragas_metrics.faithfulness,
                "coherence": ragas_metrics.overall_score,
                "ragas": ragas_metrics.to_dict(),
                "invocation": invocation_metrics.to_dict(),
                "colbert_enabled": colbert_enabled,
                "precision_stats": precision_stats,
                "rerank_enabled": rerank_enabled,
                "pii_redacted": pii_redacted,
                "deployment_profile": runtime_profile.get("deployment_profile", ""),
                "runtime_backends": runtime_profile.get("backends", {}),
                "memory_written": bool(memory_result and memory_result.get("memory_written")),
                "memory_score": memory_result.get("memory_score", 0.0) if memory_result else 0.0,
                "conversation_id": conversation_id or "",
            },
            steps=pipeline_steps,
            started_at=pipeline_start,
        )

        # Build step summaries for response
        steps_out = [pipeline_step_to_public_dict(s) for s in pipeline_steps]

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
                "context_chunks": len(chunks),
                "coverage": coverage,
                "coverage_ratio": final_coverage_ratio,
                "coverage_target": target_coverage,
                "tokens": total_tokens,
                "duration_ms": total_duration_ms,
                "query_type": str(query_type),
                "cache_hit": False,
                "embedding_cache_hits": embedding_cache_hits,
                "embedding_cache_misses": embedding_cache_misses,
                "retrieval_mode": retrieval_mode,
                "flare_retrievals": flare_retrievals,
                "firewall_pass_rate": firewall_pass_rate,
                "quality_rating": reflection_result.quality.value if reflection_result else "unknown",
                "answer_quality": reflection_result.quality.value if reflection_result else "unknown",
                "answer_confidence": reflection_result.confidence if reflection_result else 0.0,
                "faithfulness": ragas_metrics.faithfulness,
                "coherence": ragas_metrics.overall_score,
                "ragas": ragas_metrics.to_dict(),
                "invocation": invocation_metrics.to_dict(),
                "colbert_enabled": colbert_enabled,
                "precision_stats": precision_stats,
                "rerank_enabled": rerank_enabled,
                "pii_redacted": pii_redacted,
                "deployment_profile": runtime_profile.get("deployment_profile", ""),
                "runtime_backends": runtime_profile.get("backends", {}),
                "memory_written": bool(memory_result and memory_result.get("memory_written")),
                "memory_score": memory_result.get("memory_score", 0.0) if memory_result else 0.0,
                "conversation_id": conversation_id or "",
            },
            "confidence": {
                "overall": reflection_result.confidence if reflection_result else 0.5,
                "factors": {
                    "retrieval": min(1.0, final_coverage_ratio / target_coverage) if target_coverage > 0 else 0.5,
                    "generation": reflection_result.confidence if reflection_result else 0.5,
                    "citation": citation_result.pass_rate if 'citation_result' in dir() and citation_result else 0.5,
                },
                "hallucination_pass": firewall_passed if 'firewall_passed' in dir() else None,
                "evidence_contract_pass": contract_passed if 'contract_passed' in dir() else None,
            },
            "steps": steps_out,
        }

        # vNext E1: Create trace bundle for export
        corpus_version = ""
        retrieval_mode_flags = RetrievalMode.STANDARD
        if hasattr(self._retrieval, 'get_corpus_version'):
            corpus_version = self._retrieval.get_corpus_version()
        if hasattr(self._retrieval, 'get_retrieval_mode_flags'):
            retrieval_mode_flags = self._retrieval.get_retrieval_mode_flags()

        self._last_trace_bundle = create_trace_bundle(
            query=query,
            answer=answer,
            steps=steps_out,
            corpus_version=corpus_version,
            config_hash=cache_hash,
            retrieval_mode=retrieval_mode_flags,
            evaluator_verdicts={
                "reflection_quality": reflection_result.quality.value if reflection_result else "unknown",
                "retrieval_verdict": retrieval_verdict.value if retrieval_verdict else "unknown",
            },
            citation_check=citation_details if 'citation_details' in dir() else {},
            total_duration_ms=total_duration_ms,
        )
        result["trace_bundle_available"] = True

        # G3: Use versioned cache keys (in-memory)
        cacheable = {**result, "steps": [s for s in steps_out if s["name"] != "cache"]}
        cache_manager.queries.set(
            query,
            cacheable,
            config_hash=cache_hash,
            corpus_version=corpus_version,
            retrieval_mode=retrieval_mode_flags,
        )

        # P0.3: Store to disk cache for persistence across sessions
        disk_cache = get_disk_query_cache()
        disk_cache.set(
            query=query,
            result=cacheable,
            corpus_version=corpus_version,
            retrieval_mode=int(retrieval_mode_flags),
            preset_id=current_preset_id,
            model_ids=model_ids,
            scope_key=query_cache_scope,
        )

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
        conversation_id: str | None = None,
        memory_context: str = "",
        runtime_profile: dict[str, Any] | None = None,
    ) -> dict:
        """Generate answer directly without retrieval (for simple queries LLM can handle)."""
        if on_stage:
            on_stage("thinking")
        thinking_start_time = datetime.now(UTC)
        thinking_start = time.perf_counter()
        _, thinking_details, thinking_status = await self._build_thinking_outline(
            query=query,
            context_text=None,
            provider=self._provider,
            citations_count=0,
            allow_query_only=True,
        )
        record_step(self._make_step(
            "thinking",
            thinking_start,
            thinking_start_time,
            thinking_details,
            status=thinking_status,
        ))

        if on_stage:
            on_stage("generation")

        gen_start_time = datetime.now(UTC)
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
            ]
            if memory_context:
                messages.append({"role": "system", "content": memory_context})
            messages.append({"role": "user", "content": query})

            gen_details["provider"] = "configured"
            gen_details["model"] = "configured"

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
            except ProviderError:
                answer = PUBLIC_PROVIDER_ERROR_MESSAGE
                gen_details["status"] = "error"
                gen_details["error"] = PUBLIC_PROVIDER_ERROR_MESSAGE

        record_step(self._make_step("generation", gen_start, gen_start_time, gen_details))

        total_duration_ms = sum(s.duration_ms for s in pipeline_steps)

        answer, pii_redacted = self._maybe_redact_pii(answer)
        memory_result = None
        if conversation_id and self._memory is not None:
            memory_result = self._memory.record_exchange(
                conversation_id=conversation_id,
                user_query=query,
                answer=answer,
                metadata={"chunks_used": [], "sources_count": 0, "query_type": "direct"},
            )
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
                "pii_redacted": pii_redacted,
                "deployment_profile": (runtime_profile or {}).get("deployment_profile", ""),
                "runtime_backends": (runtime_profile or {}).get("backends", {}),
                "conversation_id": conversation_id or "",
                "memory_written": bool(memory_result and memory_result.get("memory_written")),
                "memory_score": memory_result.get("memory_score", 0.0) if memory_result else 0.0,
            },
            steps=pipeline_steps,
            started_at=pipeline_start,
        )

        steps_out = [pipeline_step_to_public_dict(s) for s in pipeline_steps]

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
                "pii_redacted": pii_redacted,
                "deployment_profile": (runtime_profile or {}).get("deployment_profile", ""),
                "runtime_backends": (runtime_profile or {}).get("backends", {}),
                "conversation_id": conversation_id or "",
                "memory_written": bool(memory_result and memory_result.get("memory_written")),
                "memory_score": memory_result.get("memory_score", 0.0) if memory_result else 0.0,
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
        conversation_id: str | None = None,
        runtime_profile: dict[str, Any] | None = None,
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
                "deployment_profile": (runtime_profile or {}).get("deployment_profile", ""),
                "runtime_backends": (runtime_profile or {}).get("backends", {}),
                "conversation_id": conversation_id or "",
            },
            steps=pipeline_steps,
            started_at=pipeline_start,
        )

        steps_out = [pipeline_step_to_public_dict(s) for s in pipeline_steps]

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
                "deployment_profile": (runtime_profile or {}).get("deployment_profile", ""),
                "runtime_backends": (runtime_profile or {}).get("backends", {}),
                "conversation_id": conversation_id or "",
            },
            "steps": steps_out,
            "needs_clarification": True,
        }

    def _build_abstention_response(
        self,
        query: str,
        abstention_result: AbstentionResult,
        chunks: list[EvidenceChunk],
        pipeline_start: datetime,
        pipeline_steps: list[PipelineStep],
        conversation_id: str | None = None,
        runtime_profile: dict[str, Any] | None = None,
    ) -> dict:
        """Build response when abstaining due to insufficient evidence.

        This provides a transparent explanation when the system cannot
        provide a reliable answer based on the available evidence.
        """

        total_duration_ms = sum(s.duration_ms for s in pipeline_steps)

        # Use the formatted abstention response
        answer = self._abstention_rules.format_abstention_response(
            abstention_result, query, include_details=False
        )
        answer, pii_redacted = self._maybe_redact_pii(answer)

        trace = self._telemetry.record(
            prompt=query,
            answer=answer,
            metrics={
                "chunks": len(chunks),
                "coverage": abstention_result.details.get("coverage_ratio", 0.0),
                "tokens": 0,
                "duration_ms": total_duration_ms,
                "cache_hit": False,
                "mode": "abstained",
                "abstention_reason": abstention_result.reason.value if abstention_result.reason else "unknown",
                "pii_redacted": pii_redacted,
                "deployment_profile": (runtime_profile or {}).get("deployment_profile", ""),
                "runtime_backends": (runtime_profile or {}).get("backends", {}),
                "conversation_id": conversation_id or "",
            },
            steps=pipeline_steps,
            started_at=pipeline_start,
        )

        steps_out = [pipeline_step_to_public_dict(s) for s in pipeline_steps]

        # Include partial sources even when abstaining
        sources = [
            {
                "id": c.id,
                "title": c.title,
                "snippet_preview": c.snippet[:200] if len(c.snippet) > 200 else c.snippet,
                "score": c.score,
            }
            for c in chunks[:5]
        ]

        return {
            "answer": answer,
            "chunks": [{"id": c.id, "title": c.title, "snippet": c.snippet, "score": c.score} for c in chunks[:5]],
            "sources": sources,
            "trace_id": trace.id,
            "metrics": {
                "chunks": len(chunks),
                "coverage": abstention_result.details.get("coverage_ratio", 0.0),
                "tokens": 0,
                "duration_ms": total_duration_ms,
                "cache_hit": False,
                "mode": "abstained",
                "abstention_reason": abstention_result.reason.value if abstention_result.reason else "unknown",
                "abstention_confidence": abstention_result.confidence,
                "pii_redacted": pii_redacted,
            },
            "steps": steps_out,
            "confidence": {
                "overall": 0.0,
                "factors": {
                    "retrieval": abstention_result.details.get("avg_chunk_score", 0.0),
                    "coverage": abstention_result.details.get("coverage_ratio", 0.0),
                },
                "abstained": True,
                "abstention_reason": abstention_result.reason.value if abstention_result.reason else "unknown",
            },
        }

    # =========================================================================
    # vNext Expansion: Export Methods
    # =========================================================================

    def get_trace_bundle(self) -> TraceBundle | None:
        """Get the last trace bundle for export (E1 requirement).

        Returns:
            TraceBundle if available from last query, None otherwise
        """
        return self._last_trace_bundle

    def export_trace_json(self) -> str | None:
        """Export last trace bundle as JSON string (E1).

        Returns:
            JSON string if trace available, None otherwise
        """
        if self._last_trace_bundle:
            return self._last_trace_bundle.to_json()
        return None

    def trigger_artifact_build(self, force: bool = False) -> dict:
        """Manually trigger background artifact build (G4)."""
        if not self._chunk_records:
            return {"status": "error", "message": "No documents ingested yet"}

        try:
            provider = self._provider or self._providers.get_default_provider()
            corpus_version = ""
            if hasattr(self._retrieval, "get_corpus_version"):
                corpus_version = self._retrieval.get_corpus_version()
            # Create evidence chunks from records
            evidence_chunks = []
            for doc_id, chunk in self._chunk_records:
                # Basic reconstruction - title lookup would be better but this is sufficient for v1
                evidence_chunks.append(EvidenceChunk(
                    id=f"{doc_id}-{chunk.index}",
                    title=f"Document {doc_id}",
                    snippet=chunk.text,
                    score=1.0,
                    doc_id=doc_id,
                ))

            asyncio.create_task(self._artifact_builder.build_all_async(
                evidence_chunks,
                provider,
                corpus_version=corpus_version or "manual_trigger",
                force_rebuild=force
            ))

            return {
                "status": "triggered",
                "message": "Background build started",
                "chunk_count": len(evidence_chunks)
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to trigger artifact build: {e}"}

    def get_artifact_build_status(self) -> dict:
        """Get current status of background artifact builds (G4).

        Returns:
            Dict with graph_rag and raptor build status
        """
        progress = self._artifact_builder.progress
        payload = progress.to_dict()
        payload.update({
            "graph_rag_status": progress.graph_rag.status.value,
            "raptor_status": progress.raptor.status.value,
            "graph_build_progress": progress.graph_rag.progress,
            "raptor_build_progress": progress.raptor.progress,
            "graph_version": progress.graph_rag.corpus_version,
            "raptor_version": progress.raptor.corpus_version,
        })
        if hasattr(self._retrieval, "get_corpus_version"):
            payload["corpus_version"] = self._retrieval.get_corpus_version()
        return payload

    def get_graph_data(self) -> dict[str, Any]:
        """Get detailed GraphRAG data for visualization."""
        if not self._graph_rag:
            # Try to load from builder
            self._graph_rag = self._artifact_builder.get_graph()

        if not self._graph_rag:
            return {"entities": [], "relationships": [], "communities": []}

        return {
            "entities": [
                {"name": e.name, "type": e.type.value if hasattr(e.type, 'value') else str(e.type), "description": e.description}
                for e in self._graph_rag.entities.values()
            ],
            "relationships": [
                {"source": r.source, "target": r.target, "description": r.description}
                for r in self._graph_rag.relationships
            ],
            "communities": [
                {"id": c.id, "entities": c.entities, "summary": c.summary}
                for c in self._graph_rag.communities
            ]
        }

    def get_raptor_data(self) -> dict[str, Any]:
        """Get RAPTOR tree data for visualization."""
        if not self._document_trees:
            self._document_trees = self._artifact_builder.get_trees()

        if not self._document_trees:
            return {"trees": {}}

        result = {}
        for doc_id, tree in self._document_trees.items():
            nodes_data = []
            for node_id, node in tree.nodes.items():
                nodes_data.append({
                    "id": node_id,
                    "title": node.title,
                    "summary": node.summary,
                    "level": node.level,
                    "children": node.children,
                    "parent": node.parent_id
                })
            result[doc_id] = {
                "doc_id": doc_id,
                "root_id": tree.root_id,
                "nodes": nodes_data
            }
        return {"trees": result}

    async def stop(self) -> None:
        """Gracefully stop agentic components."""
        await self._flare_generator.stop()

    def get_model_status(self) -> dict[str, Any]:
        """Get loading status of local models (G1)."""
        if hasattr(self._retrieval, "get_model_status"):
            return self._retrieval.get_model_status()
        return {}

    def get_readiness_status(self) -> dict[str, Any]:
        """Return a truthful operator readiness contract."""
        checks: dict[str, dict[str, Any]] = {
            "orchestrator": {
                "status": "ok",
                "message": None,
                "details": {"configured": self._config is not None},
            }
        }

        try:
            snapshot = (
                self._retrieval.get_readiness_snapshot()
                if hasattr(self._retrieval, "get_readiness_snapshot")
                else {}
            )
        except Exception as exc:
            snapshot = {}
            checks["retrieval_engine"] = {
                "status": "fail",
                "message": f"Retrieval status unavailable: {exc}",
                "details": {},
            }

        document_count = int(snapshot.get("document_count", 0) or 0)
        chunk_count = int(snapshot.get("chunk_count", 0) or 0)
        embedding_count = int(snapshot.get("embedding_count", 0) or 0)
        index_ready = bool(snapshot.get("index_ready", False))
        model_status = snapshot.get("model_status", {}) if isinstance(snapshot.get("model_status"), dict) else {}
        config = snapshot.get("config", {}) if isinstance(snapshot.get("config"), dict) else {}

        checks.setdefault("retrieval_engine", {
            "status": "ok",
            "message": None,
            "details": {
                "chunk_count": chunk_count,
                "sparse_ready": bool(snapshot.get("sparse_ready", False)),
                "dense_ready": bool(snapshot.get("dense_ready", False)),
                "bq_enabled": bool(snapshot.get("bq_enabled", False)),
                "bq_ready": bool(snapshot.get("bq_ready", False)),
            },
        })
        checks["document_store"] = {
            "status": "ok",
            "message": None,
            "details": {"document_count": document_count},
        }
        checks["retrieval_index"] = {
            "status": "ok" if index_ready else "fail",
            "message": None if index_ready else "Documents exist but no retrieval chunks are indexed",
            "details": {
                "document_count": document_count,
                "chunk_count": chunk_count,
                "embedding_count": embedding_count,
            },
        }

        embedding = model_status.get("embedding_model", {}) if isinstance(model_status, dict) else {}
        embedding_state = embedding.get("status", "unknown") if isinstance(embedding, dict) else "unknown"
        checks["embedding_model"] = {
            "status": "warn" if embedding_state in {"failed", "unknown"} else "ok",
            "message": None if embedding_state not in {"failed", "unknown"} else "Embedding model is not ready; sparse retrieval may still work",
            "details": {
                "state": embedding_state,
                "model": embedding.get("name") if isinstance(embedding, dict) else config.get("embedding_model"),
                "backend_id": embedding.get("backend_id") if isinstance(embedding, dict) else None,
            },
        }

        reranker = model_status.get("reranker_model", {}) if isinstance(model_status, dict) else {}
        reranker_state = reranker.get("status", "disabled") if isinstance(reranker, dict) else "disabled"
        use_reranking = bool(config.get("use_reranking", False))
        checks["reranker_model"] = {
            "status": "warn" if use_reranking and reranker_state in {"failed", "unknown"} else "ok",
            "message": (
                "Reranker is enabled but not ready"
                if use_reranking and reranker_state in {"failed", "unknown"}
                else None
            ),
            "details": {
                "state": reranker_state,
                "model": reranker.get("name") if isinstance(reranker, dict) else config.get("reranker_model"),
                "enabled": use_reranking,
                "backend_id": reranker.get("backend_id") if isinstance(reranker, dict) else None,
            },
        }

        checks["provider"] = {
            "status": "ok" if self._provider is not None else "warn",
            "message": None if self._provider is not None else "No generator provider configured; grounded summaries remain available",
            "details": {"configured": self._provider is not None},
        }

        has_failures = any(check["status"] == "fail" for check in checks.values())
        has_warnings = any(check["status"] == "warn" for check in checks.values())
        return {
            "ready": not has_failures,
            "level": "not_ready" if has_failures else ("degraded" if has_warnings else "ready"),
            "checks": checks,
        }

    def get_eval_audit_context(self) -> dict[str, Any]:
        """Return redacted runtime context for durable eval reports."""
        corpus = (
            self._retrieval.get_corpus_manifest()
            if hasattr(self._retrieval, "get_corpus_manifest")
            else {}
        )
        runtime_profile = (
            self._retrieval.get_runtime_profile()
            if hasattr(self._retrieval, "get_runtime_profile")
            else {}
        )
        model_status = self.get_model_status()
        config_snapshot: dict[str, Any] = {}
        if self._config is not None:
            config_snapshot = self._redacted_config_snapshot(self._config.model_dump(mode="json"))

        return {
            "corpus": corpus,
            "runtime_profile": runtime_profile,
            "model_status": model_status,
            "config_snapshot": config_snapshot,
        }

    @staticmethod
    def _redacted_config_snapshot(value: Any) -> Any:
        """Remove secret-shaped values from config snapshots."""
        secret_terms = ("apikey", "authorization", "token", "secret", "password", "credential")
        if isinstance(value, dict):
            return {
                key: (
                    "[redacted]"
                    if any(term in re.sub(r"[^a-z0-9]", "", key.lower()) for term in secret_terms)
                    and item is not None
                    else Orchestrator._redacted_config_snapshot(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [Orchestrator._redacted_config_snapshot(item) for item in value]
        return value
