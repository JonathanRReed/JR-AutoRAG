"""Ownership and cleanup of /query/stream producer work.

These tests drive the real FastAPI app (with its middleware stack) through a
hand-written ASGI receive/send pair so a client disconnect can be placed at an
exact point: before the first event, while the provider call is blocked, after
normal completion, after an error, and after an explicit trace cancel. The
orchestrator is a synthetic stand-in; no provider or network is contacted.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.main import app
from app.routers import query as query_router
from app.services import get_container

TIMEOUT = 5.0


class FakeOrchestrator:
    """Synthetic orchestrator whose answer() can be held, failed, or flooded."""

    def __init__(
        self,
        *,
        events_before_block: int = 0,
        block: bool = True,
        fail: str | None = None,
        trace_cancel: bool = False,
        flood_progress: int = 0,
        emit_after_cancel: int = 0,
    ) -> None:
        self.events_before_block = events_before_block
        self.block = block
        self.fail = fail
        self.trace_cancel = trace_cancel
        self.flood_progress = flood_progress
        self.emit_after_cancel = emit_after_cancel
        self.calls = 0
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()
        self.callbacks_after_cancel = 0

    async def answer(self, question, **kwargs):
        self.calls += 1
        on_stage = kwargs["on_stage"]
        on_progress = kwargs["on_progress"]
        on_token = kwargs["on_token"]
        self.started.set()
        try:
            for i in range(self.events_before_block):
                on_stage(f"stage-{i}")
                on_progress({"stage": "planning", "message": "working", "trace_id": "trace-synthetic"})
            for i in range(self.flood_progress):
                on_progress({"stage": "gatherer", "message": f"flood {i}", "trace_id": "trace-synthetic"})
            if self.block:
                # Stands in for a provider call that has not answered yet.
                await self.release.wait()
            if self.trace_cancel:
                # Same shape as Orchestrator.answer() after POST /query/cancel.
                raise asyncio.CancelledError("Trace trace-synthetic cancelled by user")
            if self.fail:
                raise RuntimeError(self.fail)
            on_token("synthetic ")
            on_token("answer")
            return {
                "answer": "synthetic answer",
                "chunks": [],
                "sources": [],
                "trace_id": "trace-synthetic",
                "metrics": {},
                "steps": [],
            }
        except asyncio.CancelledError:
            self.cancelled.set()
            # A provider SDK may still hand back buffered tokens while unwinding.
            for i in range(self.emit_after_cancel):
                on_token(f"late-{i}")
                on_progress({"stage": "late", "message": "late", "trace_id": "trace-synthetic"})
                self.callbacks_after_cancel += 1
            raise


@pytest.fixture()
def channels(monkeypatch):
    """Record every stream channel the route creates (None on pre-fix code)."""
    created: list = []
    channel_cls = getattr(query_router, "_StreamChannel", None)
    if channel_cls is not None:

        class RecordingChannel(channel_cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created.append(self)

        monkeypatch.setattr(query_router, "_StreamChannel", RecordingChannel)
    return created


def use_orchestrator(orchestrator: FakeOrchestrator) -> None:
    container = SimpleNamespace(
        orchestrator=orchestrator,
        document_store=SimpleNamespace(list=lambda: []),
    )
    app.dependency_overrides[get_container] = lambda: container


@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


class Client:
    """Minimal ASGI client that can disconnect at a chosen point."""

    def __init__(self, *, spec_version: str = "2.3", fail_send_after_disconnect: bool = False) -> None:
        self.spec_version = spec_version
        self.fail_send_after_disconnect = fail_send_after_disconnect
        self.messages: list[dict] = []
        self.body_chunks: list[bytes] = []
        self.disconnected = asyncio.Event()
        self.response_started = asyncio.Event()
        self.first_body = asyncio.Event()
        self.finished = asyncio.Event()
        self._request_sent = False

    async def receive(self):
        if not self._request_sent:
            self._request_sent = True
            body = json.dumps({"question": "synthetic stream question"}).encode()
            return {"type": "http.request", "body": body, "more_body": False}
        await self.disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(self, message):
        if self.fail_send_after_disconnect and self.disconnected.is_set():
            raise OSError("client went away")
        self.messages.append(message)
        if message["type"] == "http.response.start":
            self.response_started.set()
        elif message["type"] == "http.response.body":
            if message.get("body"):
                self.body_chunks.append(message["body"])
                self.first_body.set()
            if not message.get("more_body", False):
                self.finished.set()

    def scope(self) -> dict:
        body_len = len(json.dumps({"question": "synthetic stream question"}).encode())
        return {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": self.spec_version},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/query/stream",
            "raw_path": b"/query/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"127.0.0.1:8000"),
                (b"content-type", b"application/json"),
                (b"content-length", str(body_len).encode()),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
            "state": {},
        }

    def start(self) -> asyncio.Task:
        return asyncio.create_task(app(self.scope(), self.receive, self.send))

    def events(self) -> list[dict]:
        text = b"".join(self.body_chunks).decode()
        out = []
        for block in text.split("\n\n"):
            block = block.strip()
            if block.startswith("data:"):
                out.append(json.loads(block[len("data:"):].strip()))
        return out


def pending_producers() -> list[asyncio.Task]:
    current = asyncio.current_task()
    return [
        task
        for task in asyncio.all_tasks()
        if task is not current and not task.done() and "query-stream-producer" in task.get_name()
    ]


async def wait_for_app(task: asyncio.Task) -> None:
    try:
        await asyncio.wait_for(task, TIMEOUT)
    except (OSError, Exception) as exc:  # pragma: no cover - surfaced in asserts
        if not isinstance(exc, OSError) and type(exc).__name__ != "ClientDisconnect":
            raise


def all_tasks_except_current() -> set[asyncio.Task]:
    current = asyncio.current_task()
    return {t for t in asyncio.all_tasks() if t is not current and not t.done()}


@pytest.mark.asyncio
async def test_disconnect_before_first_event_cancels_producer(channels) -> None:
    orchestrator = FakeOrchestrator(events_before_block=0, block=True)
    use_orchestrator(orchestrator)
    before = all_tasks_except_current()
    client = Client()
    app_task = client.start()

    await asyncio.wait_for(orchestrator.started.wait(), TIMEOUT)
    assert client.events() == []
    client.disconnected.set()

    await asyncio.wait_for(orchestrator.cancelled.wait(), TIMEOUT)
    await wait_for_app(app_task)
    await asyncio.sleep(0)
    assert all_tasks_except_current() - before == set()
    assert orchestrator.release.is_set() is False


@pytest.mark.asyncio
async def test_disconnect_during_blocked_provider_cancels_producer(channels) -> None:
    orchestrator = FakeOrchestrator(events_before_block=2, block=True)
    use_orchestrator(orchestrator)
    before = all_tasks_except_current()
    client = Client()
    app_task = client.start()

    await asyncio.wait_for(client.first_body.wait(), TIMEOUT)
    assert any(event["type"] == "stage" for event in client.events())
    client.disconnected.set()

    await asyncio.wait_for(orchestrator.cancelled.wait(), TIMEOUT)
    await wait_for_app(app_task)
    await asyncio.sleep(0)
    assert all_tasks_except_current() - before == set()


@pytest.mark.asyncio
async def test_no_event_accumulation_after_disconnect(channels) -> None:
    orchestrator = FakeOrchestrator(events_before_block=1, block=True, emit_after_cancel=500)
    use_orchestrator(orchestrator)
    client = Client()
    app_task = client.start()

    await asyncio.wait_for(client.first_body.wait(), TIMEOUT)
    client.disconnected.set()
    await asyncio.wait_for(orchestrator.cancelled.wait(), TIMEOUT)
    await wait_for_app(app_task)

    # The unwinding provider kept calling back, but nothing was buffered for a
    # consumer that no longer exists.
    assert orchestrator.callbacks_after_cancel == 500
    assert len(channels) == 1
    channel = channels[0]
    assert channel.closed is True
    assert channel.buffered == 0


@pytest.mark.asyncio
async def test_flooded_queue_never_exceeds_bound_and_cleanup_does_not_deadlock(channels, monkeypatch) -> None:
    monkeypatch.setattr(query_router, "STREAM_MAX_BUFFERED_EVENTS", 16, raising=False)
    orchestrator = FakeOrchestrator(events_before_block=0, flood_progress=5000, block=True)
    use_orchestrator(orchestrator)
    before = all_tasks_except_current()
    client = Client()
    app_task = client.start()

    await asyncio.wait_for(orchestrator.started.wait(), TIMEOUT)
    await asyncio.sleep(0)
    assert len(channels) == 1
    assert channels[0].buffered <= 16 + 3
    client.disconnected.set()
    await asyncio.wait_for(orchestrator.cancelled.wait(), TIMEOUT)
    await wait_for_app(app_task)
    await asyncio.sleep(0)
    assert all_tasks_except_current() - before == set()
    assert channels[0].buffered == 0


@pytest.mark.asyncio
async def test_normal_completion_streams_result_and_releases_producer(channels) -> None:
    orchestrator = FakeOrchestrator(events_before_block=1, block=False)
    use_orchestrator(orchestrator)
    before = all_tasks_except_current()
    client = Client()
    app_task = client.start()

    await asyncio.wait_for(client.finished.wait(), TIMEOUT)
    await wait_for_app(app_task)
    events = client.events()
    assert [e["type"] for e in events if e["type"] in {"result", "error"}] == ["result"]
    assert "".join(e["data"]["text"] for e in events if e["type"] == "token") == "synthetic answer"
    assert events[-1]["data"]["answer"] == "synthetic answer"
    assert not orchestrator.cancelled.is_set()
    await asyncio.sleep(0)
    assert all_tasks_except_current() - before == set()


@pytest.mark.asyncio
async def test_error_streams_error_event_and_releases_producer(channels) -> None:
    orchestrator = FakeOrchestrator(block=False, fail="synthetic provider failure")
    use_orchestrator(orchestrator)
    before = all_tasks_except_current()
    client = Client()
    app_task = client.start()

    await asyncio.wait_for(client.finished.wait(), TIMEOUT)
    await wait_for_app(app_task)
    events = client.events()
    assert events[-1] == {"type": "error", "data": {"message": "synthetic provider failure"}}
    await asyncio.sleep(0)
    assert all_tasks_except_current() - before == set()


@pytest.mark.asyncio
async def test_explicit_trace_cancel_ends_stream_without_result_or_error(channels) -> None:
    orchestrator = FakeOrchestrator(events_before_block=1, block=True, trace_cancel=True)
    use_orchestrator(orchestrator)
    before = all_tasks_except_current()
    client = Client()
    app_task = client.start()

    await asyncio.wait_for(client.first_body.wait(), TIMEOUT)
    orchestrator.release.set()  # the gatherer sees the cancelled trace id

    await asyncio.wait_for(client.finished.wait(), TIMEOUT)
    await wait_for_app(app_task)
    types = [e["type"] for e in client.events()]
    assert "result" not in types and "error" not in types
    await asyncio.sleep(0)
    assert all_tasks_except_current() - before == set()


@pytest.mark.asyncio
async def test_spec_24_send_failure_before_start_releases_producer(channels) -> None:
    orchestrator = FakeOrchestrator(block=True)
    use_orchestrator(orchestrator)
    before = all_tasks_except_current()
    client = Client(spec_version="2.4", fail_send_after_disconnect=True)
    client.disconnected.set()  # gone before the response could start
    app_task = client.start()
    await wait_for_app(app_task)
    await asyncio.sleep(0)
    # BaseHTTPMiddleware buffers the inner response, so the body (and producer)
    # may start before the outer send fails. Either way nothing may outlive it.
    assert orchestrator.calls == 0 or orchestrator.cancelled.is_set()
    assert not orchestrator.release.is_set()
    assert pending_producers() == []
    assert all_tasks_except_current() - before == set()


@pytest.mark.asyncio
async def test_response_never_sent_never_starts_producer(channels) -> None:
    orchestrator = FakeOrchestrator(block=True)
    use_orchestrator(orchestrator)
    response = await query_router.ask_stream(
        query_router.QueryRequest(question="synthetic stream question"),
        SimpleNamespace(state=SimpleNamespace()),
        container=SimpleNamespace(orchestrator=orchestrator, document_store=SimpleNamespace(list=lambda: [])),
    )
    await asyncio.sleep(0)
    assert orchestrator.calls == 0
    assert response.producer is None
    assert pending_producers() == []


@pytest.mark.asyncio
async def test_spec_24_blocked_provider_is_released_by_keepalive_send(channels, monkeypatch) -> None:
    monkeypatch.setattr(query_router, "STREAM_KEEPALIVE_SECONDS", 0.05, raising=False)
    orchestrator = FakeOrchestrator(events_before_block=1, block=True)
    use_orchestrator(orchestrator)
    before = all_tasks_except_current()
    client = Client(spec_version="2.4", fail_send_after_disconnect=True)
    app_task = client.start()

    await asyncio.wait_for(client.first_body.wait(), TIMEOUT)
    client.disconnected.set()
    await asyncio.wait_for(orchestrator.cancelled.wait(), TIMEOUT)
    await wait_for_app(app_task)
    await asyncio.sleep(0)
    assert all_tasks_except_current() - before == set()


def test_channel_bounds_coalesces_tokens_and_finishes_when_full() -> None:
    channel_cls = query_router._StreamChannel
    channel = channel_cls(max_events=4)
    for i in range(100):
        channel.emit({"type": "token", "data": {"text": f"{i},"}})
        channel.emit({"type": "progress", "data": {"n": i}})
    assert channel.buffered <= 4
    assert channel.dropped > 0
    # Terminal delivery never blocks or raises, even with a full buffer.
    channel.finish({"type": "result", "data": {"answer": "done"}})
    items = []
    while channel.buffered:
        items.append(channel._queue.get_nowait())
    assert items[-1] is None
    assert items[-2] == {"type": "result", "data": {"answer": "done"}}
    tokens = "".join(item["data"]["text"] for item in items if item and item["type"] == "token")
    assert tokens == "".join(f"{i}," for i in range(100))


def test_closed_channel_ignores_late_callbacks() -> None:
    channel = query_router._StreamChannel(max_events=8)
    channel.emit({"type": "stage", "data": {"name": "planning"}})
    channel.close()
    assert channel.buffered == 0
    for _ in range(1000):
        channel.emit({"type": "token", "data": {"text": "late"}})
    channel.finish({"type": "result", "data": {}})
    assert channel.buffered == 0
