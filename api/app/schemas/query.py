"""Schemas for query and telemetry endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

MAX_QUESTION_LENGTH = 10000
MAX_DOCUMENT_IDS = 100


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_LENGTH, description="The question to answer")
    document_ids: list[str] | None = Field(
        default=None,
        max_length=MAX_DOCUMENT_IDS,
        description="Optional document scope for retrieval.",
    )
    history: list[dict[str, str]] | None = None
    conversation_id: str | None = None
    query_mode: Literal["grounded", "open_domain"] | None = None


class ChunkOut(BaseModel):
    id: str
    title: str
    snippet: str
    score: float


class PipelineStepOut(BaseModel):
    """A single step in the RAG pipeline with timing and details."""
    name: str
    duration_ms: float
    details: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    started_at: str | datetime | None = None
    completed_at: str | datetime | None = None


class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkOut]
    sources: list[dict[str, Any]] = Field(default_factory=list)  # Phase 2: Citation info
    trace_id: str
    metrics: dict[str, Any]  # Allow strings like query_type
    confidence: dict[str, Any] | None = None
    steps: list[PipelineStepOut] = Field(default_factory=list)
    trace_bundle_available: bool | None = None
    needs_clarification: bool | None = None


class TraceStepOut(BaseModel):
    """Step info for trace display."""
    name: str
    duration_ms: float
    details: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    started_at: str | datetime | None = None
    completed_at: str | datetime | None = None


class TraceOut(BaseModel):
    id: str
    prompt: str
    answer: str
    metrics: dict[str, Any]  # Allow strings like query_type
    steps: list[TraceStepOut] = Field(default_factory=list)
