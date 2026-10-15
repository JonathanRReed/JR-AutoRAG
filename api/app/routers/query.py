"""Query endpoints (answering questions, telemetry)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from datetime import datetime

import anyio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from ..core.auth import get_auth
from ..core.document_acl import get_acl_enforcer, resolve_acl_defaults
from ..core.query_mode import QueryMode
from ..core.telemetry import PipelineStep, pipeline_step_to_public_dict
from ..schemas.query import (
    MAX_QUESTION_LENGTH,
    QueryRequest,
    QueryResponse,
    TraceOut,
    TraceStepOut,
)
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/query", tags=["query"])


def _scoped_conversation_id(
    user_id: str | None, conversation_id: str | None
) -> str | None:
    """Return an internal, principal-bound key for conversation memory.

    Clients can choose human-readable conversation IDs, but the shared in-process
    memory store must never use that value directly.  Hash the authenticated
    principal (or the public namespace when auth is disabled) with the client ID
    so collisions across users cannot read or poison each other's memory.
    """
    if not conversation_id:
        return None
    principal = user_id or "public"
    digest = hashlib.sha256(f"{principal}:{conversation_id}".encode()).hexdigest()
    return f"conversation:{digest}"


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


# Server-sent query streams
# -------------------------
# Ownership contract for POST /query/stream:
# * The response object owns the orchestrator task. The task is created lazily
#   when the response body is first iterated, so a response that is never sent
#   never starts model work. (Behind BaseHTTPMiddleware the body can start
#   before the client-facing send; a client already gone then is handled by the
#   cleanup below.)
# * When the response ends for any reason (normal completion, error event,
#   explicit trace cancel, client disconnect, send failure, server cancellation)
#   the channel is closed and the producer task is cancelled and awaited for a
#   bounded grace period, shielded from the cancellation that ended the response.
# * Task cancellation propagates into Orchestrator.answer() at its next await.
#   Work already handed to a remote provider (and billed) may still complete on
#   the provider side; cancellation only stops this process from waiting for it.
# * Orchestrator callbacks are synchronous, so the channel never blocks them.
#   Storage for a slow or stalled consumer is finite in both events and bytes:
#   at most STREAM_MAX_BUFFERED_EVENTS non-terminal events plus one frame of
#   merged token text, and at most STREAM_MAX_BUFFERED_BYTES of encoded SSE
#   frames counting that merged text. Token text that does not fit the byte
#   budget (including one oversized token) is not stored, and neither is any
#   later token, so the streamed text is always an exact prefix of the answer.
#   A single `stream_truncated` notice then follows the prefix. Other
#   non-terminal events are dropped when either limit is reached. The final
#   `result` event carries the complete answer and steps and replaces the
#   streamed text in the client. A fixed reserve (truncation notice, merged
#   token flush, terminal event, end marker) is kept outside the limits so
#   terminal and error delivery never block or fail. Once closed, callbacks are
#   no-ops and nothing accumulates for a gone consumer.
# * While no event is ready, a comment line is sent every
#   STREAM_KEEPALIVE_SECONDS. Clients ignore it; servers on ASGI spec >= 2.4
#   (no disconnect listener in Starlette) detect a gone client on that send.
# * A producer that ignores cancellation for longer than the grace period is
#   not awaited further: the response finishes and is released. The task is
#   kept in _DETACHED_PRODUCERS until it ends, and its outcome is retrieved
#   and logged then. This process cannot force such a task to stop; that is a
#   property of the code that suppressed the cancellation.

STREAM_MAX_BUFFERED_EVENTS = 512
STREAM_MAX_BUFFERED_BYTES = 1024 * 1024
STREAM_KEEPALIVE_SECONDS = 15.0
STREAM_CANCEL_GRACE_SECONDS = 5.0
# Truncation notice, merged-token flush, terminal event, end-of-stream marker.
_STREAM_RESERVED_SLOTS = 4
_KEEPALIVE = object()
_TRUNCATED_NOTICE = {
    "type": "stream_truncated",
    "data": {
        "reason": "buffer_limit",
        "message": (
            "The live token stream stopped because the client fell behind. "
            "The final result carries the complete answer."
        ),
    },
}

logger = logging.getLogger(__name__)

# Producers that outlived their response's cancel grace period. Holding them
# here keeps a strong owner until they end, so their outcome is always observed.
_DETACHED_PRODUCERS: set[asyncio.Task[None]] = set()


def _frame_size(item: dict) -> int:
    """Encoded size of the SSE frame for ``item`` (ASCII, so chars == bytes)."""
    return len(_encode_event(item))


# "data: " + {"type": "token", "data": {"text": ""}} + "\n\n"
_TOKEN_FRAME_OVERHEAD = len(f"data: {json.dumps({'type': 'token', 'data': {'text': ''}})}\n\n")


def _escaped_size(text: str) -> int:
    """Bytes ``text`` adds inside a JSON string (json.dumps escapes per character)."""
    return len(json.dumps(text)) - 2


class _StreamChannel:
    """Bounded, closable hand-off from orchestrator callbacks to one SSE consumer."""

    def __init__(
        self,
        max_events: int = STREAM_MAX_BUFFERED_EVENTS,
        max_bytes: int = STREAM_MAX_BUFFERED_BYTES,
    ) -> None:
        self._max_events = max(1, max_events)
        self._max_bytes = max(_TOKEN_FRAME_OVERHEAD + 1, max_bytes)
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue(
            maxsize=self._max_events + _STREAM_RESERVED_SLOTS
        )
        # Byte size of each queued item, in queue order (reserved frames are 0).
        self._sizes: deque[int] = deque()
        self._queued_bytes = 0
        self._pending_tokens: list[str] = []
        self._pending_bytes = 0
        self.closed = False
        self.finished = False
        self.dropped = 0
        self.tokens_truncated = False
        self._notice_due = False

    @property
    def buffered(self) -> int:
        return self._queue.qsize()

    @property
    def buffered_bytes(self) -> int:
        """Encoded bytes held for the consumer, including merged token text."""
        pending_frame = _TOKEN_FRAME_OVERHEAD + self._pending_bytes if self._pending_tokens else 0
        return self._queued_bytes + pending_frame

    def _has_room(self) -> bool:
        return self._queue.qsize() < self._max_events

    def _put(self, item: dict | None, size: int) -> None:
        self._queue.put_nowait(item)
        self._sizes.append(size)
        self._queued_bytes += size

    def _flush_pending(self, *, use_reserved: bool = False) -> bool:
        """Queue merged token text, then a due truncation notice, in that order."""
        if self._pending_tokens:
            if not use_reserved and not self._has_room():
                return False
            text = "".join(self._pending_tokens)
            size = _TOKEN_FRAME_OVERHEAD + self._pending_bytes
            self._pending_tokens.clear()
            self._pending_bytes = 0
            self._put({"type": "token", "data": {"text": text}}, size)
        if self._notice_due:
            if not use_reserved and not self._has_room():
                return False
            self._notice_due = False
            self._put(dict(_TRUNCATED_NOTICE), 0)
        return True

    def _emit_token(self, text: str) -> None:
        if self.tokens_truncated:
            return
        cost = _escaped_size(text)
        frame = 0 if self._pending_tokens else _TOKEN_FRAME_OVERHEAD
        if self.buffered_bytes + frame + cost > self._max_bytes:
            # Storing this or any later token would break the byte bound or
            # leave a gap in the streamed text. Stop the live text here.
            self.tokens_truncated = True
            self._notice_due = True
            self._flush_pending()
            return
        self._pending_tokens.append(text)
        self._pending_bytes += cost
        self._flush_pending()

    def emit(self, item: dict) -> None:
        """Enqueue a non-terminal event without ever blocking the caller."""
        if self.closed or self.finished:
            return
        if item.get("type") == "token":
            self._emit_token(str(item.get("data", {}).get("text", "")))
            return
        # Keep order: an event cannot overtake token text still waiting for room.
        if not self._flush_pending() or not self._has_room():
            self.dropped += 1
            return
        size = _frame_size(item)
        if self.buffered_bytes + size > self._max_bytes:
            self.dropped += 1
            return
        self._put(item, size)

    def finish(self, terminal: dict | None) -> None:
        """Enqueue the terminal event (if any) and the end marker. Never blocks."""
        if self.closed or self.finished:
            return
        self.finished = True
        self._flush_pending(use_reserved=True)
        if terminal is not None:
            self._put(terminal, 0)
        self._put(None, 0)

    async def next(self, keepalive_seconds: float) -> object:
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=keepalive_seconds)
        except TimeoutError:
            return _KEEPALIVE
        if self._sizes:
            self._queued_bytes -= self._sizes.popleft()
        return item

    def close(self) -> None:
        """Stop accepting events and release anything still buffered."""
        self.closed = True
        self._pending_tokens.clear()
        self._pending_bytes = 0
        self._notice_due = False
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._sizes.clear()
        self._queued_bytes = 0


def _encode_event(item: dict) -> str:
    try:
        payload = json.dumps(item, default=_json_default)
    except Exception as exc:
        payload = json.dumps(
            {
                "type": "error",
                "data": {"message": f"Stream serialization error: {exc}"},
            },
            default=_json_default,
        )
    return f"data: {payload}\n\n"


def _observe_detached_producer(task: asyncio.Task[None]) -> None:
    _DETACHED_PRODUCERS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "detached query stream producer failed after its response ended",
            exc_info=(type(exc), exc, exc.__traceback__),
        )


class _OwnedEventStreamResponse(StreamingResponse):
    """SSE response that owns, and always releases, its producer task."""

    def __init__(
        self,
        channel: _StreamChannel,
        produce: Callable[[], Coroutine[object, object, None]],
    ) -> None:
        self._channel = channel
        self._produce = produce
        self.producer: asyncio.Task[None] | None = None
        super().__init__(self._events(), media_type="text/event-stream")

    def _start_producer(self) -> None:
        if self.producer is None and not self._channel.closed:
            self.producer = asyncio.create_task(
                self._produce(), name="query-stream-producer"
            )

    async def _events(self) -> AsyncIterator[str]:
        self._start_producer()
        try:
            while True:
                item = await self._channel.next(STREAM_KEEPALIVE_SECONDS)
                if item is _KEEPALIVE:
                    yield ": keepalive\n\n"
                    continue
                if item is None:
                    break
                yield _encode_event(item)  # type: ignore[arg-type]
        finally:
            # Synchronous part of the release; safe while being cancelled.
            self._channel.close()
            if self.producer is not None and not self.producer.done():
                self.producer.cancel()

    async def _release(self) -> None:
        self._channel.close()
        with anyio.CancelScope(shield=True):
            body = self.body_iterator
            aclose = getattr(body, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass
            producer = self.producer
            if producer is None:
                return
            if not producer.done():
                producer.cancel()
                await asyncio.wait({producer}, timeout=STREAM_CANCEL_GRACE_SECONDS)
            if not producer.done():
                # It suppressed cancellation past the grace period. Stop waiting
                # so the response is released, but keep owning the task until it
                # ends and observe whatever it ends with.
                logger.warning(
                    "query stream producer outlived its %.1fs cancel grace period; "
                    "detaching it until it finishes",
                    STREAM_CANCEL_GRACE_SECONDS,
                )
                _DETACHED_PRODUCERS.add(producer)
                producer.add_done_callback(_observe_detached_producer)
                return
            if not producer.cancelled():
                # Retrieve the outcome so it is never reported as unhandled.
                producer.exception()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._release()


def _resolve_query_access(
    payload: QueryRequest,
    request: Request,
    container: ServiceContainer,
) -> tuple[list[str] | None, str | None, str | None]:
    user_id = getattr(request.state, "user_id", None)
    scopes = getattr(request.state, "scopes", [])
    auth_enabled = get_auth().require_auth()

    if not auth_enabled:
        document_ids = payload.document_ids
        return (
            document_ids,
            _make_cache_scope(None, document_ids, payload.conversation_id),
            _scoped_conversation_id(None, payload.conversation_id),
        )

    default_public, _ = resolve_acl_defaults(auth_enabled)
    enforcer = get_acl_enforcer(default_public=default_public)

    if "admin" in scopes:
        return (
            payload.document_ids,
            _make_cache_scope(user_id, payload.document_ids, payload.conversation_id),
            _scoped_conversation_id(user_id, payload.conversation_id),
        )

    docs = container.document_store.list()

    if payload.document_ids:
        allowed = [
            doc_id
            for doc_id in payload.document_ids
            if enforcer.check_access(doc_id, user_id, "read")[0]
        ]
        if not allowed:
            raise HTTPException(
                status_code=403, detail="No access to requested documents"
            )
        return (
            allowed,
            _make_cache_scope(user_id, allowed, payload.conversation_id),
            _scoped_conversation_id(user_id, payload.conversation_id),
        )

    if not docs:
        return (
            None,
            _make_cache_scope(user_id, None, payload.conversation_id),
            _scoped_conversation_id(user_id, payload.conversation_id),
        )

    allowed_ids = [
        doc.id for doc in docs if enforcer.check_access(doc.id, user_id, "read")[0]
    ]
    if not allowed_ids:
        raise HTTPException(status_code=403, detail="No accessible documents")

    if len(allowed_ids) == len(docs):
        return (
            None,
            _make_cache_scope(user_id, None, payload.conversation_id),
            _scoped_conversation_id(user_id, payload.conversation_id),
        )
    return (
        allowed_ids,
        _make_cache_scope(user_id, allowed_ids, payload.conversation_id),
        _scoped_conversation_id(user_id, payload.conversation_id),
    )


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
    document_ids, cache_scope, memory_conversation_id = _resolve_query_access(
        payload, request, container
    )
    result = await container.orchestrator.answer(
        payload.question,
        document_ids=document_ids,
        history=payload.history,
        conversation_id=memory_conversation_id,
        query_mode=QueryMode(payload.query_mode) if payload.query_mode else None,
        cache_scope=cache_scope,
    )
    if payload.conversation_id:
        result.setdefault("metrics", {})["conversation_id"] = payload.conversation_id
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

    document_ids, cache_scope, memory_conversation_id = _resolve_query_access(
        payload, request, container
    )
    channel = _StreamChannel(
        max_events=STREAM_MAX_BUFFERED_EVENTS, max_bytes=STREAM_MAX_BUFFERED_BYTES
    )

    def on_step(step: PipelineStep) -> None:
        channel.emit({"type": "step", "data": pipeline_step_to_public_dict(step)})

    def on_token(token: str) -> None:
        channel.emit({"type": "token", "data": {"text": token}})

    def on_stage(stage: str) -> None:
        channel.emit({"type": "stage", "data": {"name": stage}})

    def on_progress(progress: dict) -> None:
        """Enhanced progress callback with human-readable messages."""
        channel.emit({"type": "progress", "data": progress})

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
                conversation_id=memory_conversation_id,
                query_mode=(
                    QueryMode(payload.query_mode) if payload.query_mode else None
                ),
                cache_scope=cache_scope,
            )
        except asyncio.CancelledError:
            channel.finish(None)
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                # The stream owner cancelled this task (client gone). Let it end
                # as cancelled so the owner's wait observes the real outcome.
                raise
            # POST /query/cancel: answer() raised CancelledError for its trace
            # while this task itself was not cancelled. The stream ends with no
            # result and no error event, as before.
            return
        except Exception as exc:
            channel.finish({"type": "error", "data": {"message": str(exc)}})
            return
        if payload.conversation_id:
            result.setdefault("metrics", {})["conversation_id"] = payload.conversation_id
        channel.finish({"type": "result", "data": result})

    return _OwnedEventStreamResponse(channel, run_query)


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
                TraceStepOut(**pipeline_step_to_public_dict(s)) for s in trace.steps
            ],
        )
        for trace in traces
    ]


@router.post("/cancel")
async def cancel_trace(
    trace_id: str, container: ServiceContainer = Depends(get_container)
):
    container.orchestrator.cancel_trace(trace_id)
    return {"status": "cancelled", "trace_id": trace_id}
