"""Experiment schemas for local AutoRAG runs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalMetricResult(BaseModel):
    name: str
    value: float
    provider: str = "local_heuristic"
    direction: str = "higher_is_better"
    details: dict[str, object] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    name: str = Field(default="Local quality matrix", min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    parser: list[str] = Field(default_factory=lambda: ["native", "docling"], max_length=8)
    chunker: list[str] = Field(default_factory=lambda: ["recursive"], max_length=8)
    embedding: list[str] = Field(default_factory=list, max_length=8)
    dense_weight: list[float] = Field(default_factory=lambda: [0.55, 0.65, 0.75], max_length=8)
    sparse_weight: list[float] = Field(default_factory=lambda: [0.45, 0.35, 0.25], max_length=8)
    reranker: list[bool] = Field(default_factory=lambda: [True, False], max_length=4)
    graph: list[bool] = Field(default_factory=lambda: [False, True], max_length=4)
    raptor: list[bool] = Field(default_factory=lambda: [False, True], max_length=4)
    ocr_policy: list[str] = Field(default_factory=lambda: ["auto"], max_length=8)
    questions: list[str] = Field(default_factory=list, max_length=50)


class ExperimentRun(BaseModel):
    id: str
    config: ExperimentConfig
    status: str
    created_at: str
    owner_id: str = "anonymous"
    completed_at: str | None = None
    metrics: list[EvalMetricResult] = Field(default_factory=list)
    winning_preset: str | None = None
    config_snapshot: dict[str, object] = Field(default_factory=dict)
    traces: list[str] = Field(default_factory=list)
    promoted_at: str | None = None


class ExperimentPromoteResponse(BaseModel):
    run: ExperimentRun
    promoted_preset: str
