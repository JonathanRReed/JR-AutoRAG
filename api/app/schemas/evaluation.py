"""Evaluation-related schemas."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel

from .query import QueryResponse


class EvaluationRequest(BaseModel):
    name: str
    questions: List[str]


class EvaluationRun(BaseModel):
    name: str
    responses: List[QueryResponse]
    average_coverage: float
    average_tokens: float
