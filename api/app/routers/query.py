"""Query endpoints (answering questions, telemetry)."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..core.telemetry import PipelineStep
from ..schemas.query import QueryRequest, QueryResponse, TraceOut, TraceStepOut
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def ask(payload: QueryRequest, container: ServiceContainer = Depends(get_container)):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = await container.orchestrator.answer(payload.question, document_ids=payload.document_ids)
    return QueryResponse(**result)


@router.post("/stream")
async def ask_stream(payload: QueryRequest, container: ServiceContainer = Depends(get_container)):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def serialize_step(step: PipelineStep) -> dict:
        return {
            "name": step.name,
            "duration_ms": step.duration_ms,
            "details": step.details,
            "status": step.status,
            "started_at": step.started_at.isoformat(),
            "completed_at": step.completed_at.isoformat(),
        }

    def on_step(step: PipelineStep) -> None:
        queue.put_nowait({"type": "step", "data": serialize_step(step)})

    def on_token(token: str) -> None:
        queue.put_nowait({"type": "token", "data": {"text": token}})

    def on_stage(stage: str) -> None:
        queue.put_nowait({"type": "stage", "data": {"name": stage}})

    async def run_query() -> None:
        try:
            result = await container.orchestrator.answer(
                payload.question,
                document_ids=payload.document_ids,
                on_step=on_step,
                on_token=on_token,
                on_stage=on_stage,
            )
            await queue.put({"type": "result", "data": result})
        except Exception as exc:
            await queue.put({"type": "error", "data": {"message": str(exc)}})
        finally:
            await queue.put(None)

    asyncio.create_task(run_query())

    async def event_stream():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
