"""Regression tests for exposed API authentication defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.core import auth as auth_module
from app.core.security_middleware import verify_api_key


REPO_ROOT = Path(__file__).resolve().parents[2]


def _reset_auth() -> None:
    auth_module._auth_instance = None


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "server": ("testserver", 80),
            "scheme": "http",
            "client": ("203.0.113.10", 4444),
        }
    )


def test_docker_image_enables_exposed_authenticated_mode() -> None:
    dockerfile = (REPO_ROOT / "api" / "Dockerfile").read_text()

    assert "AUTORAG_EXPOSE=true" in dockerfile
    assert "AUTORAG_AUTH_ENABLED=true" in dockerfile


def test_compose_requires_api_keys_for_published_api() -> None:
    compose = (REPO_ROOT / "docker-compose.yml").read_text()

    assert 'AUTORAG_EXPOSE: "true"' in compose
    assert 'AUTORAG_AUTH_ENABLED: "true"' in compose
    assert "AUTORAG_API_KEYS: ${AUTORAG_API_KEYS:?" in compose


@pytest.mark.asyncio
async def test_config_route_rejects_missing_api_key_when_auth_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTORAG_AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTORAG_API_KEYS", "test-admin-key")
    monkeypatch.delenv("AUTORAG_EXPOSE", raising=False)
    _reset_auth()

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(_request("/config"), api_key=None)

    assert exc_info.value.status_code == 401

    _reset_auth()


@pytest.mark.asyncio
async def test_exposed_mode_fails_closed_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTORAG_EXPOSE", "true")
    monkeypatch.setenv("AUTORAG_AUTH_ENABLED", "false")
    monkeypatch.delenv("AUTORAG_API_KEYS", raising=False)
    _reset_auth()

    with pytest.raises(HTTPException) as exc_info:
        await verify_api_key(_request("/documents"), api_key=None)

    assert exc_info.value.status_code == 503

    _reset_auth()
