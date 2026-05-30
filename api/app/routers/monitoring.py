"""Monitoring and telemetry APIs."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..core.cache import get_cache_manager
from ..core.telemetry import pipeline_step_to_public_dict
from ..schemas.query import TraceOut, TraceStepOut
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/traces", response_model=list[TraceOut])
def traces(container: ServiceContainer = Depends(get_container)):
    traces = container.telemetry.list()
    return [
        TraceOut(
            id=trace.id,
            prompt=trace.prompt,
            answer=trace.answer,
            metrics=trace.metrics,
            steps=[
                TraceStepOut(**pipeline_step_to_public_dict(s))
                for s in trace.steps
            ],
        )
        for trace in traces
    ]


@router.get("/cache")
def cache_stats():
    cache = get_cache_manager()
    return cache.stats()


@router.post("/cache/clear")
def cache_clear():
    cache = get_cache_manager()
    cache.clear_all()
    return cache.stats()


# ============================================================================
# Observability Dashboard Endpoints (Tier 2)
# ============================================================================

@router.get("/metrics")
def get_metrics(container: ServiceContainer = Depends(get_container)):
    """Get comprehensive metrics for dashboard display.

    Returns aggregated metrics including:
    - Query counts and latencies (avg, p50, p95)
    - Cache hit rates
    - Quality distribution
    - Stage latency breakdown
    - FLARE trigger rate
    - Hallucination pass rate
    """
    return container.telemetry.get_full_metrics_export()


@router.get("/metrics/stages")
def get_stage_metrics(container: ServiceContainer = Depends(get_container)):
    """Get per-stage latency breakdown.

    Useful for identifying bottlenecks in the RAG pipeline.
    Returns average duration (ms) per stage.
    """
    return {
        "stages": container.telemetry.get_stage_latency_breakdown(),
        "percentiles": container.telemetry.get_stage_latency_percentiles(),
    }


@router.get("/metrics/retrieval")
def get_retrieval_metrics(container: ServiceContainer = Depends(get_container)):
    """Get retrieval-specific metrics.

    Includes mode distribution (standard, RAPTOR, GraphRAG),
    rerank usage rate, and FLARE trigger rate.
    """
    return {
        "mode_distribution": container.telemetry.get_retrieval_mode_distribution(),
        "rerank_usage_rate": container.telemetry._calculate_rerank_usage(),
        "flare_trigger_rate": container.telemetry.get_flare_trigger_rate(),
    }


@router.get("/metrics/quality")
def get_quality_metrics(container: ServiceContainer = Depends(get_container)):
    """Get answer quality metrics.

    Includes quality distribution and hallucination firewall pass rate.
    """
    return {
        "quality_distribution": container.telemetry._calculate_quality_distribution(),
        "hallucination_pass_rate": container.telemetry.get_hallucination_pass_rate(),
    }


@router.get("/health/detailed")
def detailed_health(container: ServiceContainer = Depends(get_container)):
    """Get detailed system health status.

    Checks all components and returns their status.
    """
    cache = get_cache_manager()
    cache_stats = cache.stats()

    return {
        "status": "healthy",
        "components": {
            "telemetry": {
                "status": "ok",
                "trace_count": len(container.telemetry.list()),
            },
            "embedding_cache": {
                "status": "ok",
                "size": cache_stats["embeddings"]["size"],
                "hit_rate": cache_stats["embeddings"]["hit_rate"],
            },
            "query_cache": {
                "status": "ok",
                "size": cache_stats["queries"]["size"],
                "hit_rate": cache_stats["queries"]["hit_rate"],
            },
            "models": container.orchestrator.get_model_status(),
        },
    }
