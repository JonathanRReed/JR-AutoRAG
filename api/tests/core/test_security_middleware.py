"""Tests for route-to-scope authorization decisions."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core import security_middleware
from app.core.auth import APIKeyAuth
from app.core.security_middleware import _resolve_required_scope, verify_api_key


@pytest.mark.parametrize(
    ("path", "method", "scope"),
    [
        ("/monitoring/traces", "GET", "read"),
        ("/api/traces/last", "GET", "read"),
        ("/api/artifacts/graph", "GET", "read"),
        ("/api/cache/status", "GET", "read"),
        ("/api/metrics/presets/estimates", "GET", "read"),
        ("/providers/local", "GET", "read"),
        ("/api/artifacts/build", "POST", "admin"),
        ("/api/cache/clear", "DELETE", "admin"),
        ("/api/cache/rebuild", "POST", "admin"),
        ("/api/metrics/presets/clear", "DELETE", "admin"),
        ("/monitoring/cache/clear", "POST", "admin"),
    ],
)
def test_sensitive_protected_routes_resolve_expected_scopes(path: str, method: str, scope: str) -> None:
    assert _resolve_required_scope(path, method) == scope


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
async def test_verify_api_key_fails_closed_for_unmapped_protected_route(monkeypatch) -> None:
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
