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


class ParsedBlock(BaseModel):
    type: str
    text: str = ""
    page: int | None = None
    heading_level: int | None = None
    confidence: float = 1.0
    metadata: dict[str, object] = Field(default_factory=dict)


class ParsedPage(BaseModel):
    number: int
    text: str = ""
    confidence: float = 1.0
    metadata: dict[str, object] = Field(default_factory=dict)
    blocks: list[ParsedBlock] = Field(default_factory=list)


class DocumentPreviewResponse(BaseModel):
    document_id: str
    title: str
    parser_provider: str
    parser_engine: str
    confidence: float = 0.0
    used_ocr: bool = False
    page_count: int = 0
    block_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    blocks: list[ParsedBlock] = Field(default_factory=list)
    pages: list[ParsedPage] = Field(default_factory=list)
