"""Monitoring and telemetry APIs."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..schemas.query import TraceOut
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/traces", response_model=list[TraceOut])
def traces(container: ServiceContainer = Depends(get_container)):
    traces = container.telemetry.list()
    return [TraceOut(id=trace.id, prompt=trace.prompt, answer=trace.answer, metrics=trace.metrics) for trace in traces]
