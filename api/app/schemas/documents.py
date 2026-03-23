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
    sync: bool = False
    ocr_policy: str | None = None
    langextract_profile_override: str | None = None
    langextract_prompt_override: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    title: str
    chunk_count: int
