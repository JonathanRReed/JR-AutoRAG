"""Tests for route-to-scope authorization decisions."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.responses import PlainTextResponse

from app.core import security_middleware
from app.core.auth import APIKeyAuth
from app.core.security_middleware import _resolve_required_scope, verify_api_key


@pytest.mark.parametrize(
    ("path", "method", "scope"),
    [
        ("/monitoring/traces", "GET", "admin"),
        ("/api/traces/last", "GET", "admin"),
        ("/query/traces", "GET", "admin"),
        ("/query/cancel", "POST", "admin"),
        ("/api/artifacts/graph", "GET", "read"),
        ("/api/cache/status", "GET", "read"),
        ("/api/metrics/presets/estimates", "GET", "read"),
        ("/providers/local", "GET", "admin"),
        ("/api/artifacts/build", "POST", "admin"),
        ("/api/cache/clear", "DELETE", "admin"),
        ("/api/cache/rebuild", "POST", "admin"),
        ("/api/metrics/presets/clear", "DELETE", "admin"),
        ("/monitoring/cache/clear", "POST", "admin"),
    ],
)
def test_sensitive_protected_routes_resolve_expected_scopes(
    path: str, method: str, scope: str
) -> None:
    assert _resolve_required_scope(path, method) == scope


def test_provider_helpers_require_admin_scope_to_use_configured_credentials() -> None:
    auth = APIKeyAuth(enabled=True)
    read_key, _ = auth.generate_key("reader", scopes=["read"])
    write_key, _ = auth.generate_key("writer", scopes=["write"])
    admin_key, _ = auth.generate_key("admin", scopes=["admin"])

    required_scope = _resolve_required_scope("/providers/openrouter/test", "POST")

    assert required_scope == "admin"
    assert not auth.verify(read_key, required_scope=required_scope)
    assert not auth.verify(write_key, required_scope=required_scope)
    assert auth.verify(admin_key, required_scope=required_scope)


def test_evaluation_scope_resolution_is_method_specific() -> None:
    assert _resolve_required_scope("/evaluation/golden-sets", "GET") == "read"
    assert _resolve_required_scope("/evaluation/runs", "GET") == "read"
    assert _resolve_required_scope("/evaluation/runs/run-1/report", "GET") == "admin"
    assert _resolve_required_scope("/evaluation", "POST") == "eval"
    assert _resolve_required_scope("/evaluation/golden-sets", "POST") == "eval"
    assert _resolve_required_scope("/evaluation/golden-sets/demo", "DELETE") == "eval"
    assert _resolve_required_scope("/evaluation/batch/demo", "POST") == "eval"


def test_evaluation_mutations_require_eval_or_admin_scope() -> None:
    auth = APIKeyAuth(enabled=True)
    read_key, _ = auth.generate_key("reader", scopes=["read"])
    write_key, _ = auth.generate_key("writer", scopes=["write"])
    eval_key, _ = auth.generate_key("evaluator", scopes=["eval"])
    admin_key, _ = auth.generate_key("admin", scopes=["admin"])

    required_scope = _resolve_required_scope("/evaluation/golden-sets", "POST")

    assert not auth.verify(read_key, required_scope=required_scope)
    assert not auth.verify(write_key, required_scope=required_scope)
    assert auth.verify(eval_key, required_scope=required_scope)
    assert auth.verify(admin_key, required_scope=required_scope)


def test_more_specific_scope_mapping_wins_for_mutating_routes() -> None:
    read_key_auth = APIKeyAuth(enabled=True)
    read_key, _ = read_key_auth.generate_key("reader", scopes=["read"])

    assert not read_key_auth.verify(
        read_key,
        required_scope=_resolve_required_scope("/monitoring/cache/clear", "POST"),
    )
    assert not read_key_auth.verify(
        read_key,
        required_scope=_resolve_required_scope("/api/metrics/presets/clear", "DELETE"),
    )


@pytest.mark.asyncio
async def test_verify_api_key_fails_closed_for_unmapped_protected_route(
    monkeypatch,
) -> None:
    auth = APIKeyAuth(enabled=True)
    api_key, _ = auth.generate_key("reader", scopes=["read"])

    class URL:
        path = "/unmapped/protected"

    class State:
        pass

    class Client:
        host = "127.0.0.1"

    class Request:
        url = URL()
        method = "GET"
        state = State()
        client = Client()

    class AuditLog:
        def log_auth(self, **kwargs) -> None:
            pass

    monkeypatch.setattr(security_middleware, "get_auth", lambda: auth)
    monkeypatch.setattr(security_middleware, "get_audit_log", lambda: AuditLog())

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(Request(), api_key=api_key)

    assert exc_info.value.status_code == 403
    assert "not configured" in exc_info.value.detail


async def _run_http_middleware(
    middleware_type,
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    body_chunks: list[bytes] | None = None,
) -> tuple[list[dict], bool]:
    downstream_called = False
    chunks = list(body_chunks or [b""])
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive() -> dict:
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    async def downstream(scope: dict, receive_call, send_call) -> None:
        nonlocal downstream_called
        downstream_called = True
        while True:
            message = await receive_call()
            if message["type"] != "http.request" or not message.get("more_body", False):
                break
        await PlainTextResponse("ok")(scope, receive_call, send_call)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/documents/upload",
        "raw_path": b"/documents/upload",
        "query_string": b"",
        "headers": headers or [],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    await middleware_type(downstream)(scope, receive, send)
    return sent, downstream_called


@pytest.mark.asyncio
async def test_request_size_limit_counts_streamed_bytes_not_only_header(
    monkeypatch,
) -> None:
    monkeypatch.setattr(security_middleware, "MAX_REQUEST_SIZE", 4)

    sent, downstream_called = await _run_http_middleware(
        security_middleware.RequestSizeLimitMiddleware,
        headers=[(b"content-length", b"1")],
        body_chunks=[b"abc", b"def"],
    )

    response_start = next(
        message for message in sent if message["type"] == "http.response.start"
    )
    assert response_start["status"] == 413
    assert downstream_called is False


@pytest.mark.asyncio
async def test_unsafe_cross_origin_request_is_rejected_before_route_runs() -> None:
    middleware_type = security_middleware.UnsafeOriginGuardMiddleware

    sent, downstream_called = await _run_http_middleware(
        middleware_type,
        headers=[(b"origin", b"https://attacker.example")],
    )

    response_start = next(
        message for message in sent if message["type"] == "http.response.start"
    )
    assert response_start["status"] == 403
    assert downstream_called is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"origin", b"http://localhost:3000")],
    ],
)
async def test_unsafe_request_allows_non_browser_and_trusted_local_origin(
    headers,
) -> None:
    middleware_type = security_middleware.UnsafeOriginGuardMiddleware

    sent, downstream_called = await _run_http_middleware(
        middleware_type, headers=headers
    )

    response_start = next(
        message for message in sent if message["type"] == "http.response.start"
    )
    assert response_start["status"] == 200
    assert downstream_called is True
