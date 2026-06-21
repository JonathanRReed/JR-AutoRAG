"""Document ingestion and listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ..core.auth import get_auth
from ..core.cache import get_cache_manager
from ..core.document_acl import get_acl_enforcer, get_acl_store, resolve_acl_defaults
from ..core.document_parser import build_preview_from_document_metadata
from ..core.persistence import get_disk_query_cache
from ..schemas.documents import DocumentOut, DocumentPreviewResponse, IngestResponse, IngestTextRequest
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/documents", tags=["documents"])


def _invalidate_query_caches_after_document_mutation() -> None:
    """Prevent stale query responses from exposing deleted document content."""
    get_cache_manager().queries.invalidate_all()
    get_disk_query_cache().clear()


def _ensure_document_read_access(document_id: str, request: Request) -> None:
    auth_enabled = get_auth().require_auth()
    if not auth_enabled:
        return
    scopes = getattr(request.state, "scopes", [])
    if "admin" in scopes:
        return
    default_public, _ = resolve_acl_defaults(auth_enabled)
    enforcer = get_acl_enforcer(default_public=default_public)
    user_id = getattr(request.state, "user_id", None)
    allowed, _ = enforcer.check_access(document_id, user_id, "read")
    if not allowed:
        raise HTTPException(status_code=403, detail="Insufficient permissions to read this document")


@router.get("", response_model=list[DocumentOut])
def list_documents(
    request: Request,
    container: ServiceContainer = Depends(get_container),
):
    docs = container.document_store.list()
    auth_enabled = get_auth().require_auth()
    if not auth_enabled:
        return [DocumentOut(id=doc.id, title=doc.title, text=doc.text, metadata=doc.metadata) for doc in docs]

    scopes = getattr(request.state, "scopes", [])
    if "admin" in scopes:
        return [DocumentOut(id=doc.id, title=doc.title, text=doc.text, metadata=doc.metadata) for doc in docs]

    default_public, _ = resolve_acl_defaults(auth_enabled)
    enforcer = get_acl_enforcer(default_public=default_public)
    user_id = getattr(request.state, "user_id", None)
    allowed_docs = [
        doc for doc in docs if enforcer.check_access(doc.id, user_id, "read")[0]
    ]
    return [
        DocumentOut(id=doc.id, title=doc.title, text=doc.text, metadata=doc.metadata)
        for doc in allowed_docs
    ]


@router.get("/{document_id}/preview", response_model=DocumentPreviewResponse)
def get_document_preview(
    document_id: str,
    request: Request,
    container: ServiceContainer = Depends(get_container),
):
    doc = container.document_store.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    _ensure_document_read_access(document_id, request)
    payload = build_preview_from_document_metadata(doc.metadata, doc.text)
    return DocumentPreviewResponse(
        document_id=doc.id,
        title=doc.title,
        parser_provider=str(payload.get("parser_provider", "native")),
        parser_engine=str(payload.get("parser_engine", "stored-text")),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        used_ocr=bool(payload.get("used_ocr", False)),
        page_count=int(payload.get("page_count", 0) or 0),
        block_count=int(payload.get("block_count", 0) or 0),
        warnings=list(payload.get("warnings", []) or []),
        blocks=list(payload.get("blocks", []) or []),
        pages=list(payload.get("pages", []) or []),
    )


@router.post("/text", response_model=IngestResponse)
def ingest_text(
    payload: IngestTextRequest,
    request: Request,
    container: ServiceContainer = Depends(get_container),
):
    auth_enabled = get_auth().require_auth()
    scopes = getattr(request.state, "scopes", [])
    user_id = getattr(request.state, "user_id", None)
    default_public, new_doc_public = resolve_acl_defaults(auth_enabled)
    enforcer = get_acl_enforcer(default_public=default_public)

    if auth_enabled and "admin" not in scopes:
        existing = container.document_store.get_by_title(payload.title)
        if existing and not enforcer.check_access(existing.id, user_id, "write")[0]:
            raise HTTPException(status_code=403, detail="Insufficient permissions to update this document")

    try:
        result = container.ingest.ingest_text(
            title=payload.title,
            text=payload.text,
            metadata={
                **(payload.metadata or {}),
                **({"ocr_policy": payload.ocr_policy} if payload.ocr_policy else {}),
            },
            sync=payload.sync,
            langextract_profile_override=payload.langextract_profile_override,
            langextract_prompt_override=payload.langextract_prompt_override,
        )
        if enforcer.store.get(result.document_id) is None:
            owner_id = user_id or "anonymous"
            public = new_doc_public if (auth_enabled and user_id) else True
            enforcer.create_acl_for_document(result.document_id, owner=owner_id, public=public)
        return IngestResponse(document_id=result.document_id, title=result.title, chunk_count=result.chunk_count)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    request: Request,
    container: ServiceContainer = Depends(get_container),
):
    doc = container.document_store.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    auth_enabled = get_auth().require_auth()
    scopes = getattr(request.state, "scopes", [])
    if auth_enabled and "admin" not in scopes:
        user_id = getattr(request.state, "user_id", None)
        default_public, _ = resolve_acl_defaults(auth_enabled)
        enforcer = get_acl_enforcer(default_public=default_public)
        allowed, _ = enforcer.check_access(document_id, user_id, "write")
        if not allowed:
            raise HTTPException(status_code=403, detail="Insufficient permissions to delete this document")

    container.document_store.delete(document_id)
    get_acl_store().delete(document_id)
    container.retrieval_engine.build()
    _invalidate_query_caches_after_document_mutation()


@router.delete("", status_code=204)
def delete_all_documents(
    request: Request,
    container: ServiceContainer = Depends(get_container),
):
    auth_enabled = get_auth().require_auth()
    scopes = getattr(request.state, "scopes", [])
    if auth_enabled and "admin" not in scopes:
        raise HTTPException(status_code=403, detail="Admin scope required to delete all documents")

    container.document_store.clear()
    get_acl_store().clear()
    container.retrieval_engine.build()
    _invalidate_query_caches_after_document_mutation()


@router.post("/upload", response_model=IngestResponse)
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    title: str = Form(...),
    sync: bool = Form(False),
    ocr_policy: str | None = Form(None),
    langextract_profile_override: str | None = Form(None),
    langextract_prompt_override: str | None = Form(None),
    container: ServiceContainer = Depends(get_container),
):
    auth_enabled = get_auth().require_auth()
    scopes = getattr(request.state, "scopes", [])
    user_id = getattr(request.state, "user_id", None)
    default_public, new_doc_public = resolve_acl_defaults(auth_enabled)
    enforcer = get_acl_enforcer(default_public=default_public)
    filename = file.filename or "untitled"
    effective_title = title or filename

    if auth_enabled and "admin" not in scopes:
        existing = container.document_store.get_by_title(effective_title)
        if existing and not enforcer.check_access(existing.id, user_id, "write")[0]:
            raise HTTPException(status_code=403, detail="Insufficient permissions to update this document")

    try:
        content = await file.read()
        result = container.ingest.ingest_file(
            title=effective_title,
            content=content,
            metadata={
                "filename": filename,
                "content_type": file.content_type or "application/octet-stream",
                **({"ocr_policy": ocr_policy} if ocr_policy else {}),
            },
            sync=sync,
            langextract_profile_override=langextract_profile_override,
            langextract_prompt_override=langextract_prompt_override,
        )
        if enforcer.store.get(result.document_id) is None:
            owner_id = user_id or "anonymous"
            public = new_doc_public if (auth_enabled and user_id) else True
            enforcer.create_acl_for_document(result.document_id, owner=owner_id, public=public)
        return IngestResponse(document_id=result.document_id, title=result.title, chunk_count=result.chunk_count)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
