"""Onboarding and disposable demo endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.auth import get_auth
from ..core.document_acl import get_acl_enforcer, get_acl_store, resolve_acl_defaults
from ..core.onboarding import create_onboarding_flow, get_example_queries, get_sample_documents
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


def _demo_metadata(tags: list[str], order: int) -> dict[str, str]:
    return {
        "source": "demo",
        "demo_corpus": "true",
        "demo_order": str(order),
        "tags": ",".join(tags),
        "retention": "disposable",
    }


def _create_acl(document_id: str, request: Request) -> None:
    auth_enabled = get_auth().require_auth()
    user_id = getattr(request.state, "user_id", None)
    default_public, new_doc_public = resolve_acl_defaults(auth_enabled)
    enforcer = get_acl_enforcer(default_public=default_public)
    if enforcer.store.get(document_id) is None:
        public = new_doc_public if (auth_enabled and user_id) else True
        enforcer.create_acl_for_document(document_id, owner=user_id or "anonymous", public=public)


def _ensure_document_write_access(document_id: str, request: Request) -> None:
    auth_enabled = get_auth().require_auth()
    if not auth_enabled:
        return
    scopes = getattr(request.state, "scopes", [])
    if "admin" in scopes:
        return
    default_public, _ = resolve_acl_defaults(auth_enabled)
    enforcer = get_acl_enforcer(default_public=default_public)
    user_id = getattr(request.state, "user_id", None)
    allowed, _ = enforcer.check_access(document_id, user_id, "write")
    if not allowed:
        raise HTTPException(status_code=403, detail="Insufficient permissions to update this document")


@router.get("")
def get_onboarding(container: ServiceContainer = Depends(get_container)) -> dict:
    """Return first-run checklist, demo corpus metadata, and example prompts."""
    docs = container.document_store.list()
    demo_docs = [doc for doc in docs if doc.metadata.get("demo_corpus") == "true"]
    return {
        "flow": create_onboarding_flow().to_dict(),
        "demo_mode": container.demo_mode,
        "demo_seeded": len(demo_docs) > 0,
        "document_count": len(docs),
        "demo_document_count": len(demo_docs),
        "sample_documents": [
            {
                "title": item["title"],
                "tags": item.get("tags", []),
                "demo_question": item.get("demo_question", ""),
            }
            for item in get_sample_documents()
        ],
        "example_queries": get_example_queries(),
    }


@router.post("/demo/seed")
def seed_demo_corpus(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> dict:
    """Seed an evaluator-friendly local demo corpus."""
    seeded: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []

    for index, item in enumerate(get_sample_documents(), start=1):
        title = str(item["title"])
        existing = container.document_store.get_by_title(title)
        if existing:
            _ensure_document_write_access(existing.id, request)
            existing.metadata.update(_demo_metadata(list(item.get("tags", [])), index))
            container.document_store.upsert(existing)
            _create_acl(existing.id, request)
            skipped.append({"id": existing.id, "title": existing.title})
            continue

        result = container.ingest.ingest_text(
            title=title,
            text=str(item["content"]),
            metadata=_demo_metadata(list(item.get("tags", [])), index),
            sync=True,
        )
        _create_acl(result.document_id, request)
        seeded.append({"id": result.document_id, "title": result.title})

    return {
        "seeded": seeded,
        "skipped": skipped,
        "document_count": len(container.document_store.list()),
        "example_queries": get_example_queries(),
        "demo_mode": container.demo_mode,
    }


@router.delete("/demo")
def clear_demo_corpus(
    request: Request,
    container: ServiceContainer = Depends(get_container),
) -> dict:
    """Remove only demo-tagged documents."""
    demo_docs = [
        doc for doc in list(container.document_store.list())
        if doc.metadata.get("demo_corpus") == "true"
    ]
    for doc in demo_docs:
        _ensure_document_write_access(doc.id, request)

    deleted = 0
    for doc in demo_docs:
        container.document_store.delete(doc.id)
        get_acl_store().delete(doc.id)
        deleted += 1
    if deleted:
        container.retrieval_engine.build()
    return {"deleted": deleted, "document_count": len(container.document_store.list())}
