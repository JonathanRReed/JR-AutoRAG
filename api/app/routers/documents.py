"""Document ingestion and listing endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..schemas.documents import DocumentOut, IngestTextRequest, IngestResponse
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentOut])
def list_documents(container: ServiceContainer = Depends(get_container)):
    docs = container.document_store.list()
    return [DocumentOut(id=doc.id, title=doc.title, text=doc.text, metadata=doc.metadata) for doc in docs]


@router.post("/text", response_model=IngestResponse)
def ingest_text(payload: IngestTextRequest, container: ServiceContainer = Depends(get_container)):
    try:
        result = container.ingest.ingest_text(title=payload.title, text=payload.text, metadata=payload.metadata)
        return IngestResponse(document_id=result.document_id, title=result.title, chunk_count=result.chunk_count)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, container: ServiceContainer = Depends(get_container)):
    if not container.document_store.get(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    container.document_store.delete(document_id)
    container.retrieval_engine.build()


@router.post("/upload", response_model=IngestResponse)
async def ingest_file(
    title: str,
    file: UploadFile = File(...),
    container: ServiceContainer = Depends(get_container),
):
    try:
        content = await file.read()
        result = container.ingest.ingest_file(
            title=title or file.filename,
            content=content,
            metadata={"filename": file.filename},
        )
        return IngestResponse(document_id=result.document_id, title=result.title, chunk_count=result.chunk_count)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
