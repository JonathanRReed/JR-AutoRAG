"""Schemas for query and telemetry endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    document_ids: list[str] | None = None


class ChunkOut(BaseModel):
    id: str
    title: str
    snippet: str
    score: float


class PipelineStepOut(BaseModel):
    """A single step in the RAG pipeline with timing and details."""
    name: str
    duration_ms: float
    details: dict[str, Any] = {}
    status: str = "completed"
    started_at: str | None = None
    completed_at: str | None = None


class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkOut]
    sources: list[dict[str, Any]] = []  # Phase 2: Citation info
    trace_id: str
    metrics: dict[str, Any]  # Allow strings like query_type
    steps: list[PipelineStepOut] = []


class TraceStepOut(BaseModel):
    """Step info for trace display."""
    name: str
    duration_ms: float
    details: dict[str, Any] = {}
    status: str = "completed"
    started_at: str | None = None
    completed_at: str | None = None


class TraceOut(BaseModel):
    id: str
    prompt: str
    answer: str
    metrics: dict[str, Any]  # Allow strings like query_type
    steps: list[TraceStepOut] = []
