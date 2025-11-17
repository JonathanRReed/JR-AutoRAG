"""Document- and ingestion-related schemas."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: str
    title: str
    text: str
    metadata: Dict[str, str] = Field(default_factory=dict)


class IngestTextRequest(BaseModel):
    title: str
    text: str
    metadata: Optional[Dict[str, str]] = None


class IngestResponse(BaseModel):
    document_id: str
    title: str
    chunk_count: int
