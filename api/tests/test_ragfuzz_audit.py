"""Regression tests for durable RAGFuzz audit logging."""

from __future__ import annotations

import pytest

from app.routers import ragfuzz_audit


class FakeOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def ingest(self, **kwargs: object) -> None:
        self.calls.append(("ingest", kwargs))

    async def answer(self, query: str, document_ids: list[str] | None = None) -> dict[str, str]:
        self.calls.append(("answer", {"query": query, "document_ids": document_ids}))
        return {"answer": "response with CANARY_TEST_TOKEN"}

    async def delete_document(self, document_id: str) -> None:
        self.calls.append(("delete", document_id))


class FakeContainer:
    def __init__(self) -> None:
        self.orchestrator = FakeOrchestrator()


class FailingAuditLog:
    def __init__(self) -> None:
        self.entries: list[object] = []

    def log(self, entry: object) -> None:
        self.entries.append(entry)
        raise OSError("audit write failed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "payload", "expected_call"),
    [
        (
            ragfuzz_audit.inject_poison_document,
            ragfuzz_audit.PoisonDocumentRequest(content="poison", canary_token="CANARY_TEST_TOKEN"),
            "ingest",
        ),
        (
            ragfuzz_audit.check_canary_leak,
            ragfuzz_audit.CanaryCheckRequest(query="check", canary_token="CANARY_TEST_TOKEN"),
            "answer",
        ),
    ],
)
async def test_ragfuzz_audit_write_failure_prevents_success_response(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: object,
    payload: object,
    expected_call: str,
) -> None:
    """Sensitive RAGFuzz endpoints must not return success before the audit write finishes."""
    audit_log = FailingAuditLog()
    container = FakeContainer()
    monkeypatch.setattr(ragfuzz_audit, "get_audit_log", lambda: audit_log)

    with pytest.raises(OSError, match="audit write failed"):
        await endpoint(request=payload, container=container)

    assert container.orchestrator.calls[0][0] == expected_call
    assert len(audit_log.entries) == 1


@pytest.mark.asyncio
async def test_ragfuzz_delete_audit_write_failure_prevents_success_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Poison deletion must not return success before the audit delete entry is durable."""
    audit_log = FailingAuditLog()
    container = FakeContainer()
    monkeypatch.setattr(ragfuzz_audit, "get_audit_log", lambda: audit_log)

    with pytest.raises(OSError, match="audit write failed"):
        await ragfuzz_audit.remove_poison_document("poison_doc", container=container)

    assert container.orchestrator.calls == [("delete", "poison_doc")]
    assert len(audit_log.entries) == 1

@pytest.mark.asyncio
async def test_ragfuzz_health_audit_storage_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Health check should return 500 if audit storage is unavailable."""
    class FailingQueryAuditLog:
        def query(self, *args, **kwargs):
            raise OSError("audit read failed")

    audit_log = FailingQueryAuditLog()
    container = FakeContainer()
    monkeypatch.setattr(ragfuzz_audit, "get_audit_log", lambda: audit_log)
    monkeypatch.setattr(ragfuzz_audit, "_is_production_env", lambda: False)

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await ragfuzz_audit.ragfuzz_health(container=container)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Ragfuzz audit storage unavailable"
