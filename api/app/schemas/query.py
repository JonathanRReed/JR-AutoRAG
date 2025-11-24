"""Schemas for query and telemetry endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


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


class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkOut]
    trace_id: str
    metrics: dict[str, float]
    steps: list[PipelineStepOut] = []


class TraceStepOut(BaseModel):
    """Step info for trace display."""
    name: str
    duration_ms: float
    details: dict[str, Any] = {}
    status: str = "completed"


class TraceOut(BaseModel):
    id: str
    prompt: str
    answer: str
    metrics: dict[str, float]
    steps: list[TraceStepOut] = []
