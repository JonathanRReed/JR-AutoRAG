"""Evaluation-related schemas."""

from __future__ import annotations

from pydantic import BaseModel

from .query import QueryResponse


class EvaluationRequest(BaseModel):
    name: str
    questions: list[str]


class EvaluationRun(BaseModel):
    name: str
    responses: list[QueryResponse]
    average_coverage: float
    average_tokens: float
