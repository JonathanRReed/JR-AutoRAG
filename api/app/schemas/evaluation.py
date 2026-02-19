"""Evaluation-related schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .query import QueryResponse


class EvaluationRequest(BaseModel):
    name: str
    questions: list[str]


class EvaluationRun(BaseModel):
    name: str
    responses: list[QueryResponse]
    average_coverage: float
    average_tokens: float


# ============================================================================
# Golden Set Evaluation Schemas
# ============================================================================

class GoldenTestCaseSchema(BaseModel):
    """A single test case with expected results."""
    question: str
    expected_source_ids: list[str] = Field(default_factory=list)
    expected_answer_points: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    id: str | None = None


class GoldenSetCreateRequest(BaseModel):
    """Request to create a golden test set."""
    name: str
    cases: list[GoldenTestCaseSchema]


class GoldenSetInfo(BaseModel):
    """Summary info about a golden set."""
    name: str
    count: int


class RetrievalMetricsSchema(BaseModel):
    """Retrieval quality metrics."""
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    citation_coverage: float = 0.0


class AnswerMetricsSchema(BaseModel):
    """Answer quality metrics."""
    faithfulness: float = 0.0
    completeness: float = 0.0
    refusal_accuracy: float = 0.0
    coherence: float = 0.0


class TestCaseResultSchema(BaseModel):
    """Result of evaluating a single test case."""
    test_case_id: str
    question: str
    answer: str
    retrieved_source_ids: list[str]
    retrieval_metrics: RetrievalMetricsSchema
    answer_metrics: AnswerMetricsSchema
    duration_ms: float = 0.0
    trace_id: str = ""


class EvalRunResultSchema(BaseModel):
    """Result of a complete evaluation run."""
    run_id: str
    golden_set_name: str
    timestamp: datetime
    retrieval_metrics: RetrievalMetricsSchema
    answer_metrics: AnswerMetricsSchema
    individual_results: list[TestCaseResultSchema] = Field(default_factory=list)
    duration_ms: float = 0.0


class EvalRunSummary(BaseModel):
    """Summary of an eval run for listing."""
    run_id: str
    golden_set_name: str
    timestamp: str
    retrieval_metrics: RetrievalMetricsSchema
    answer_metrics: AnswerMetricsSchema
    duration_ms: float = 0.0


class RunComparisonResult(BaseModel):
    """Result of comparing two evaluation runs."""
    run_a: str
    run_b: str
    retrieval: dict
    answer: dict
    regressions: list[str]
