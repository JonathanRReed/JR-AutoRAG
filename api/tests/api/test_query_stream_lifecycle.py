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


# --- Byte-bounded storage under a token flood --------------------------------


def stored_bytes(channel) -> int:
    """Bytes the channel is holding, measured from its storage, not its counters.

    Queued events are counted at their encoded SSE frame size; merged token
    text that has not been queued yet is counted by length. This deliberately
    does not trust ``buffered`` (a queue size) or ``buffered_bytes``.
    """
    queued = 0
    for item in list(channel._queue._queue):
        if isinstance(item, dict):
            queued += len(query_router._encode_event(item))
    pending = sum(len(text) for text in channel._pending_tokens)
    return queued + pending


class TokenFloodOrchestrator:
    """Emits many large, numbered tokens with yields in between, then answers."""

    def __init__(self, *, tokens: int, token_size: int, probe) -> None:
        self.tokens = [f"<{i:05d}>" + "x" * token_size for i in range(tokens)]
        self.probe = probe
        self.flooded = asyncio.Event()
        self.started = asyncio.Event()

    async def answer(self, question, **kwargs):
        on_token = kwargs["on_token"]
        kwargs["on_stage"]("generating")
        self.started.set()
        for i, token in enumerate(self.tokens):
            on_token(token)
            self.probe()
            if i % 50 == 0:
                await asyncio.sleep(0)
        self.flooded.set()
        answer = "".join(self.tokens)
        return {"answer": answer, "chunks": [], "sources": [], "trace_id": "trace-flood", "metrics": {}, "steps": []}


class StalledClient(Client):
    """A connected client that stops reading after its first body chunk."""

    def __init__(self) -> None:
        super().__init__()
        self.resume = asyncio.Event()

    async def send(self, message):
        await super().send(message)
        if message["type"] == "http.response.body" and message.get("body") and len(self.body_chunks) == 1:
            await self.resume.wait()


@pytest.mark.asyncio
async def test_token_flood_with_stalled_consumer_stays_within_byte_bound(channels, monkeypatch) -> None:
    max_bytes = 64 * 1024
    monkeypatch.setattr(query_router, "STREAM_MAX_BUFFERED_EVENTS", 16, raising=False)
    monkeypatch.setattr(query_router, "STREAM_MAX_BUFFERED_BYTES", max_bytes, raising=False)
    peak = {"bytes": 0}

    def probe() -> None:
        peak["bytes"] = max(peak["bytes"], stored_bytes(channels[0]))

    orchestrator = TokenFloodOrchestrator(tokens=2000, token_size=1000, probe=probe)
    use_orchestrator(orchestrator)
    before = all_tasks_except_current()
    client = StalledClient()
    app_task = client.start()

    await asyncio.wait_for(orchestrator.flooded.wait(), TIMEOUT)
    # About 2 MB of tokens were produced while the consumer was stalled.
    assert peak["bytes"] <= max_bytes, f"stored {peak['bytes']} bytes for a stalled consumer"

    client.resume.set()
    await asyncio.wait_for(client.finished.wait(), TIMEOUT)
    await wait_for_app(app_task)
    events = client.events()
    full = "".join(orchestrator.tokens)
    streamed = "".join(e["data"]["text"] for e in events if e["type"] == "token")
    types = [e["type"] for e in events]
    # Honest partial stream: an exact prefix, a truncation notice, then the
    # complete final result.
    assert full.startswith(streamed) and len(streamed) < len(full)
    assert types.count("stream_truncated") == 1
    assert types[-1] == "result"
    assert types.index("stream_truncated") < types.index("result")
    assert all(t != "token" for t in types[types.index("stream_truncated"):])
    assert events[-1]["data"]["answer"] == full
    await asyncio.sleep(0)
    assert all_tasks_except_current() - before == set()


def drain(channel) -> list:
    items = []
    while channel.buffered:
        items.append(channel._queue.get_nowait())
    return items


def test_oversized_single_token_is_not_stored_and_stream_stays_an_honest_prefix() -> None:
    channel = query_router._StreamChannel(max_events=8, max_bytes=1024)
    channel.emit({"type": "token", "data": {"text": "a"}})
    channel.emit({"type": "token", "data": {"text": "b" * 10_000}})
    assert stored_bytes(channel) <= 1024
    # Later text would leave a gap after the dropped token, so it is not sent.
    channel.emit({"type": "token", "data": {"text": "c"}})
    channel.emit({"type": "stage", "data": {"name": "finalizing"}})
    assert stored_bytes(channel) <= 1024
    channel.finish({"type": "result", "data": {"answer": "a" + "b" * 10_000 + "c"}})
    items = drain(channel)
    assert items[-1] is None
    types = [item["type"] for item in items[:-1]]
    assert "".join(i["data"]["text"] for i in items[:-1] if i["type"] == "token") == "a"
    assert types.count("stream_truncated") == 1
    assert types[-1] == "result"
    assert items[-2]["data"]["answer"].endswith("c")


def test_coalesced_pending_tokens_count_against_the_byte_budget() -> None:
    channel = query_router._StreamChannel(max_events=2, max_bytes=2000)
    channel.emit({"type": "progress", "data": {"n": 1}})
    channel.emit({"type": "progress", "data": {"n": 2}})
    # The event slots are full, so tokens merge into pending text.
    for i in range(50):
        channel.emit({"type": "token", "data": {"text": f"{i:03d}" + "y" * 97}})
        assert stored_bytes(channel) <= 2000
    assert channel.buffered_bytes <= 2000
    channel.finish({"type": "error", "data": {"message": "provider failed"}})
    items = drain(channel)
    assert items[-1] is None
    assert items[-2] == {"type": "error", "data": {"message": "provider failed"}}
    streamed = "".join(i["data"]["text"] for i in items[:-1] if i["type"] == "token")
    full = "".join(f"{i:03d}" + "y" * 97 for i in range(50))
    assert streamed and full.startswith(streamed) and streamed != full
    assert [i["type"] for i in items[:-1]].count("stream_truncated") == 1


@pytest.mark.asyncio
async def test_live_consumer_receives_every_token_in_order_without_truncation() -> None:
    channel = query_router._StreamChannel(max_events=8)
    received: list[dict] = []
    for i in range(300):
        channel.emit({"type": "token", "data": {"text": f"t{i};"}})
        if i % 5 == 0:
            channel.emit({"type": "progress", "data": {"n": i}})
        if i % 3 == 0:
            while channel.buffered:
                received.append(await channel.next(1.0))
    channel.finish({"type": "result", "data": {"answer": "ok"}})
    while True:
        item = await channel.next(1.0)
        if item is None:
            break
        received.append(item)
    tokens = "".join(e["data"]["text"] for e in received if e["type"] == "token")
    assert tokens == "".join(f"t{i};" for i in range(300))
    assert all(e["type"] != "stream_truncated" for e in received)
    assert received[-1] == {"type": "result", "data": {"answer": "ok"}}


# --- A producer that outlives the cancel grace period ------------------------


@pytest.mark.asyncio
async def test_cancellation_resistant_producer_is_owned_and_observed_after_grace(monkeypatch, caplog) -> None:
    import gc
    import logging

    monkeypatch.setattr(query_router, "STREAM_CANCEL_GRACE_SECONDS", 0.05, raising=False)
    loop = asyncio.get_running_loop()
    unhandled: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    started = asyncio.Event()
    allow_exit = asyncio.Event()
    resisted = {"count": 0}

    async def stubborn_producer() -> None:
        started.set()
        while not allow_exit.is_set():
            try:
                await allow_exit.wait()
            except asyncio.CancelledError:
                # Stands in for a provider SDK that swallows cancellation.
                resisted["count"] += 1
        raise RuntimeError("late provider failure after grace")

    channel = query_router._StreamChannel(max_events=8)
    response = query_router._OwnedEventStreamResponse(channel, stubborn_producer)
    client = Client()
    try:
        with caplog.at_level(logging.WARNING, logger=query_router.__name__):
            call = asyncio.create_task(response(client.scope(), client.receive, client.send))
            await asyncio.wait_for(started.wait(), TIMEOUT)
            client.disconnected.set()
            # Response cleanup is bounded by the grace period, not by the producer.
            await asyncio.wait_for(call, 2.0)
            producer = response.producer
            assert producer is not None and not producer.done()
            assert resisted["count"] >= 1

            allow_exit.set()
            for _ in range(20):
                await asyncio.sleep(0)
            assert producer.done()
            del producer, response, call
            gc.collect()
            await asyncio.sleep(0)
        never_retrieved = [c for c in unhandled if "never retrieved" in str(c.get("message", ""))]
        assert never_retrieved == []
        messages = [record.getMessage() for record in caplog.records]
        assert any("outlived" in m for m in messages), messages
        assert any("late provider failure after grace" in (record.exc_text or str(record.exc_info)) for record in caplog.records)
        assert getattr(query_router, "_DETACHED_PRODUCERS", set()) == set()
    finally:
        loop.set_exception_handler(previous_handler)
