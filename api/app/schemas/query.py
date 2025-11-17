"""Schemas for query and telemetry endpoints."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class ChunkOut(BaseModel):
    id: str
    title: str
    snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    chunks: List[ChunkOut]
    trace_id: str
    metrics: Dict[str, float]


class TraceOut(BaseModel):
    id: str
    prompt: str
    answer: str
    metrics: Dict[str, float]
