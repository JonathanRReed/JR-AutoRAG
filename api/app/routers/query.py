"""Query endpoints (answering questions, telemetry)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..core.auth import get_auth
from ..core.document_acl import get_acl_enforcer, resolve_acl_defaults
from ..core.telemetry import PipelineStep
from ..schemas.query import MAX_QUESTION_LENGTH, QueryRequest, QueryResponse, TraceOut, TraceStepOut
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/query", tags=["query"])


def _make_cache_scope(
    user_id: str | None,
    document_ids: list[str] | None,
    conversation_id: str | None = None,
) -> str | None:
    if not user_id and not document_ids and not conversation_id:
        return None
    parts = [user_id or "public"]
    if document_ids:
        parts.append(",".join(sorted(document_ids)))
    if conversation_id:
        parts.append(f"conversation:{conversation_id}")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return digest[:16]


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _resolve_query_access(
    payload: QueryRequest,
    request: Request,
    container: ServiceContainer,
) -> tuple[list[str] | None, str | None]:
    user_id = getattr(request.state, "user_id", None)
    scopes = getattr(request.state, "scopes", [])
    auth_enabled = get_auth().require_auth()

    if not auth_enabled:
        document_ids = payload.document_ids
        return document_ids, _make_cache_scope(None, document_ids, payload.conversation_id)

    default_public, _ = resolve_acl_defaults(auth_enabled)
    enforcer = get_acl_enforcer(default_public=default_public)

    if "admin" in scopes:
        return payload.document_ids, _make_cache_scope(user_id, payload.document_ids, payload.conversation_id)

    docs = container.document_store.list()

    if payload.document_ids:
        allowed = [
            doc_id
            for doc_id in payload.document_ids
            if enforcer.check_access(doc_id, user_id, "read")[0]
        ]
        if not allowed:
            raise HTTPException(status_code=403, detail="No access to requested documents")
        return allowed, _make_cache_scope(user_id, allowed, payload.conversation_id)

    if not docs:
        return None, _make_cache_scope(user_id, None, payload.conversation_id)

    allowed_ids = [
        doc.id for doc in docs if enforcer.check_access(doc.id, user_id, "read")[0]
    ]
    if not allowed_ids:
        raise HTTPException(status_code=403, detail="No accessible documents")

    if len(allowed_ids) == len(docs):
        return None, _make_cache_scope(user_id, None, payload.conversation_id)
    return allowed_ids, _make_cache_scope(user_id, allowed_ids, payload.conversation_id)


@router.post("", response_model=QueryResponse)
async def ask(
    payload: QueryRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(payload.question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters",
        )
    document_ids, cache_scope = _resolve_query_access(payload, request, container)
    result = await container.orchestrator.answer(
        payload.question,
        document_ids=document_ids,
        history=payload.history,
        conversation_id=payload.conversation_id,
        cache_scope=cache_scope,
    )
    return QueryResponse(**result)


@router.post("/stream")
async def ask_stream(
    payload: QueryRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(payload.question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters",
        )

    document_ids, cache_scope = _resolve_query_access(payload, request, container)
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

    def on_progress(progress: dict) -> None:
        """Enhanced progress callback with human-readable messages."""
        queue.put_nowait({"type": "progress", "data": progress})

    async def run_query() -> None:
        try:
            result = await container.orchestrator.answer(
                payload.question,
                document_ids=document_ids,
                on_step=on_step,
                on_token=on_token,
                on_stage=on_stage,
                on_progress=on_progress,
                history=payload.history,
                conversation_id=payload.conversation_id,
                cache_scope=cache_scope,
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
            try:
                payload = json.dumps(item, default=_json_default)
            except Exception as exc:
                payload = json.dumps(
                    {"type": "error", "data": {"message": f"Stream serialization error: {exc}"}},
                    default=_json_default,
                )
            yield f"data: {payload}\n\n"

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
@router.post("/cancel")
async def cancel_trace(trace_id: str, container: ServiceContainer = Depends(get_container)):
    container.orchestrator.cancel_trace(trace_id)
    return {"status": "cancelled", "trace_id": trace_id}
