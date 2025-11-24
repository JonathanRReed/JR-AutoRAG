"""Query endpoints (answering questions, telemetry)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.query import QueryRequest, QueryResponse, TraceOut, TraceStepOut
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def ask(payload: QueryRequest, container: ServiceContainer = Depends(get_container)):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = await container.orchestrator.answer(payload.question)
    return QueryResponse(**result)


@router.get("/traces", response_model=list[TraceOut])
def list_traces(container: ServiceContainer = Depends(get_container)):
    traces = container.telemetry.list()
    return [
        TraceOut(
            id=trace.id,
            prompt=trace.prompt,
            answer=trace.answer,
            metrics=trace.metrics,
            steps=[
                TraceStepOut(name=s.name, duration_ms=s.duration_ms, details=s.details, status=s.status)
                for s in trace.steps
            ],
        )
        for trace in traces
    ]
