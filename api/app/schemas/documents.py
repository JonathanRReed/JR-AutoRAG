"""Document- and ingestion-related schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    title: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestTextRequest(BaseModel):
    title: str
    text: str
    metadata: dict[str, str] | None = None


class IngestResponse(BaseModel):
    document_id: str
    title: str
    chunk_count: int
