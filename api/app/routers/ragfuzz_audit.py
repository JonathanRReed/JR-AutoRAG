"""RAGFuzz audit endpoints for grey-box tracing and security testing.

These endpoints provide hooks for RAGFuzz (or similar fuzzing tools) to:
1. Inject poisoned documents for testing retrieval robustness
2. Trace pipeline execution for grey-box analysis
3. Run canary leak detection tests
4. Query for vulnerable document patterns

Environment variables:
- AUTORAG_RAGFUZZ_ENABLED: Enable RAGFuzz endpoints (default: true in dev, false in prod)
- AUTORAG_RAGFUZZ_SECRET: Optional shared secret for RAGFuzz authentication
"""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from ..core.audit import AuditAction, AuditEntry, get_audit_log
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/rag/audit", tags=["ragfuzz"])


def _is_production_env() -> bool:
    env = (
        os.environ.get("AUTORAG_ENV")
        or os.environ.get("ENVIRONMENT")
        or os.environ.get("NODE_ENV")
        or ""
    ).strip().lower()
    return env in {"prod", "production"}


def _flag_enabled(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"true", "1", "yes", "on"}


def _check_ragfuzz_enabled() -> None:
    """Check if RAGFuzz endpoints are enabled.

    Defaults:
    - Development: enabled
    - Production: disabled
    """
    default_enabled = not _is_production_env()
    enabled = _flag_enabled(os.environ.get("AUTORAG_RAGFUZZ_ENABLED"), default_enabled)
    if not enabled:
        raise HTTPException(
            status_code=403,
            detail="RAGFuzz endpoints are disabled. Set AUTORAG_RAGFUZZ_ENABLED=true to enable.",
        )


def _verify_ragfuzz_secret(x_ragfuzz_secret: str | None = Header(None)) -> None:
    """Verify RAGFuzz shared secret if configured."""
    expected = os.environ.get("AUTORAG_RAGFUZZ_SECRET")
    if _is_production_env() and not expected:
        raise HTTPException(
            status_code=503,
            detail="AUTORAG_RAGFUZZ_SECRET is required in production.",
        )
    if expected and (not x_ragfuzz_secret or not secrets.compare_digest(x_ragfuzz_secret, expected)):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-RagFuzz-Secret header",
        )


class PoisonDocumentRequest(BaseModel):
    """Request to inject a poisoned document for testing."""
    content: str = Field(..., description="Poisoned document content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    canary_token: str | None = Field(None, description="Optional canary token to embed")
    poison_type: str = Field("canary", description="Type: canary, adversarial, injection")


class PoisonDocumentResponse(BaseModel):
    """Response after injecting a poisoned document."""
    document_id: str
    canary_token: str
    injected_at: str
    content_hash: str


class CanaryCheckRequest(BaseModel):
    """Request to check if a canary was leaked."""
    query: str = Field(..., description="Query to test for canary leakage")
    canary_token: str = Field(..., description="Canary token to look for")
    document_ids: list[str] | None = Field(None, description="Limit to specific documents")


class CanaryCheckResponse(BaseModel):
    """Result of canary leak detection."""
    leaked: bool
    canary_token: str
    answer: str
    leak_score: float = Field(..., description="0.0-1.0 confidence of leak")
    matched_fragments: list[str] = Field(default_factory=list)


class TraceRequest(BaseModel):
    """Request for grey-box pipeline tracing."""
    query: str
    document_ids: list[str] | None = None
    trace_depth: str = Field("full", description="Trace depth: minimal, standard, full")


class TraceResponse(BaseModel):
    """Grey-box trace of pipeline execution."""
    trace_id: str
    query: str
    answer: str
    stages: list[dict[str, Any]]
    retrieval_details: dict[str, Any]
    timing_ms: dict[str, float]
    token_counts: dict[str, int]


class HealthResponse(BaseModel):
    """RAGFuzz health check response."""
    status: str
    ragfuzz_enabled: bool
    corpus_size: int
    providers_available: list[str]
    version: str


@router.get("/health", response_model=HealthResponse)
async def ragfuzz_health(
    container: ServiceContainer = Depends(get_container),
) -> HealthResponse:
    """Health check endpoint for RAGFuzz integration."""
    _check_ragfuzz_enabled()
    default_enabled = not _is_production_env()
    enabled = _flag_enabled(os.environ.get("AUTORAG_RAGFUZZ_ENABLED"), default_enabled)

    corpus_size = 0
    try:
        if container.orchestrator and hasattr(container.orchestrator, "doc_store"):
            docs = container.orchestrator.doc_store.list() if container.orchestrator.doc_store else []
            corpus_size = len(docs)
    except Exception:
        pass

    providers = []
    try:
        from ..core.providers import discover_local_providers
        local = await discover_local_providers()
        providers = [p.name for p in local if p.status == "ok"]
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        ragfuzz_enabled=enabled,
        corpus_size=corpus_size,
        providers_available=providers,
        version="1.0.0",
    )


@router.post("/poison", response_model=PoisonDocumentResponse)
async def inject_poison_document(
    request: PoisonDocumentRequest,
    container: ServiceContainer = Depends(get_container),
    _: None = Depends(_verify_ragfuzz_secret),
) -> PoisonDocumentResponse:
    """Inject a poisoned document for testing retrieval robustness.

    This endpoint allows security testing tools to inject adversarial
    content and verify the RAG pipeline handles it correctly.
    """
    _check_ragfuzz_enabled()

    canary = request.canary_token or f"CANARY_{uuid.uuid4().hex[:12].upper()}"
    content_with_canary = f"{request.content}\n\n<!-- canary:{canary} -->"

    content_hash = hashlib.sha256(content_with_canary.encode()).hexdigest()[:16]
    doc_id = f"poison_{content_hash}"

    metadata = {
        **request.metadata,
        "_ragfuzz_poison": True,
        "_poison_type": request.poison_type,
        "_canary_token": canary,
        "_injected_at": datetime.utcnow().isoformat(),
    }

    try:
        if container.orchestrator and hasattr(container.orchestrator, "ingest"):
            await container.orchestrator.ingest(
                content=content_with_canary,
                filename=f"poison_{doc_id}.txt",
                metadata=metadata,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to inject document: {exc}") from exc

    audit_log = get_audit_log()
    audit_log.log(AuditEntry(
        timestamp=datetime.utcnow(),
        action=AuditAction.INGEST,
        details={
            "ragfuzz": True,
            "poison_type": request.poison_type,
            "document_id": doc_id,
            "canary_token": canary,
        },
    ))

    return PoisonDocumentResponse(
        document_id=doc_id,
        canary_token=canary,
        injected_at=datetime.utcnow().isoformat(),
        content_hash=content_hash,
    )


@router.post("/canary-check", response_model=CanaryCheckResponse)
async def check_canary_leak(
    request: CanaryCheckRequest,
    container: ServiceContainer = Depends(get_container),
    _: None = Depends(_verify_ragfuzz_secret),
) -> CanaryCheckResponse:
    """Check if a canary token was leaked in an answer.

    Runs a query and checks if the canary token appears in the response,
    indicating potential data leakage or prompt injection vulnerability.
    """
    _check_ragfuzz_enabled()

    try:
        result = await container.orchestrator.answer(
            request.query,
            document_ids=request.document_ids,
        )
        answer = result.get("answer", "")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    canary_lower = request.canary_token.lower()
    answer_lower = answer.lower()

    leaked = canary_lower in answer_lower

    matched_fragments = []
    if leaked:
        idx = answer_lower.find(canary_lower)
        start = max(0, idx - 50)
        end = min(len(answer), idx + len(request.canary_token) + 50)
        matched_fragments.append(answer[start:end])

    leak_score = 1.0 if leaked else 0.0
    if not leaked:
        canary_parts = request.canary_token.split("_")
        partial_matches = sum(1 for part in canary_parts if part.lower() in answer_lower)
        leak_score = partial_matches / max(len(canary_parts), 1) * 0.5

    audit_log = get_audit_log()
    audit_log.log(AuditEntry(
        timestamp=datetime.utcnow(),
        action=AuditAction.QUERY,
        details={
            "ragfuzz": True,
            "test_type": "canary_check",
            "leaked": leaked,
            "leak_score": leak_score,
        },
    ))

    return CanaryCheckResponse(
        leaked=leaked,
        canary_token=request.canary_token,
        answer=answer,
        leak_score=leak_score,
        matched_fragments=matched_fragments,
    )


@router.post("/trace", response_model=TraceResponse)
async def trace_pipeline(
    request: TraceRequest,
    container: ServiceContainer = Depends(get_container),
    _: None = Depends(_verify_ragfuzz_secret),
) -> TraceResponse:
    """Run a query with full grey-box tracing.

    Returns detailed information about each pipeline stage for
    security analysis and fuzzing feedback.
    """
    _check_ragfuzz_enabled()

    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    stages: list[dict[str, Any]] = []
    timing_ms: dict[str, float] = {}
    token_counts: dict[str, int] = {}

    def on_step(step: Any) -> None:
        stages.append({
            "name": step.name,
            "duration_ms": step.duration_ms,
            "status": step.status,
            "details": step.details,
        })
        timing_ms[step.name] = step.duration_ms

    try:
        result = await container.orchestrator.answer(
            request.query,
            document_ids=request.document_ids,
            on_step=on_step,
        )
        answer = result.get("answer", "")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trace failed: {exc}") from exc

    retrieval_details: dict[str, Any] = {}
    if "documents" in result:
        retrieval_details["document_count"] = len(result["documents"])
        retrieval_details["document_ids"] = [d.get("id", "") for d in result.get("documents", [])]
    if "sources" in result:
        retrieval_details["sources"] = result["sources"]

    token_counts["answer"] = len(answer.split())
    token_counts["context"] = sum(
        len(str(d.get("content", "")).split())
        for d in result.get("documents", [])
    )

    return TraceResponse(
        trace_id=trace_id,
        query=request.query,
        answer=answer,
        stages=stages,
        retrieval_details=retrieval_details,
        timing_ms=timing_ms,
        token_counts=token_counts,
    )


@router.delete("/poison/{document_id}")
async def remove_poison_document(
    document_id: str,
    container: ServiceContainer = Depends(get_container),
    _: None = Depends(_verify_ragfuzz_secret),
) -> dict[str, str]:
    """Remove a previously injected poison document."""
    _check_ragfuzz_enabled()

    try:
        if container.orchestrator and hasattr(container.orchestrator, "delete_document"):
            await container.orchestrator.delete_document(document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to remove document: {exc}") from exc

    audit_log = get_audit_log()
    audit_log.log(AuditEntry(
        timestamp=datetime.utcnow(),
        action=AuditAction.DELETE,
        details={
            "ragfuzz": True,
            "document_id": document_id,
        },
    ))

    return {"status": "deleted", "document_id": document_id}


@router.get("/poison")
async def list_poison_documents(
    container: ServiceContainer = Depends(get_container),
    _: None = Depends(_verify_ragfuzz_secret),
) -> list[dict[str, Any]]:
    """List all injected poison documents."""
    _check_ragfuzz_enabled()

    poison_docs = []
    try:
        if container.orchestrator and hasattr(container.orchestrator, "doc_store"):
            docs = container.orchestrator.doc_store.list() if container.orchestrator.doc_store else []
            for doc in docs:
                meta = getattr(doc, "metadata", {}) or {}
                if meta.get("_ragfuzz_poison"):
                    poison_docs.append({
                        "document_id": getattr(doc, "id", ""),
                        "poison_type": meta.get("_poison_type"),
                        "canary_token": meta.get("_canary_token"),
                        "injected_at": meta.get("_injected_at"),
                    })
    except Exception:
        pass

    return poison_docs
