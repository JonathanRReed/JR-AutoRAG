"""Monitoring and telemetry APIs."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..core.cache import get_cache_manager
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
                TraceStepOut(
                    name=s.name,
                    duration_ms=s.duration_ms,
                    details=s.details,
                    status=s.status,
                    started_at=s.started_at.isoformat(),
                    completed_at=s.completed_at.isoformat(),
                )
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
