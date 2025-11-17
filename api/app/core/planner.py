"""Planner service that produces retrieval configurations per query."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..schemas.config import AppConfig


@dataclass
class PlanStep:
    query: str
    dense_k: int
    sparse_k: int
    rerank_pool: int
    compression: bool


@dataclass
class RetrievalPlan:
    steps: List[PlanStep]
    target_tokens: int
    coverage_target: float


class Planner:
    """Simple planner that uses AppConfig defaults."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def rebuild(self, config: AppConfig) -> None:
        self._config = config

    def plan(self, query: str) -> RetrievalPlan:
        defaults = self._config.retrieval
        step = PlanStep(
            query=query,
            dense_k=defaults.dense_k,
            sparse_k=defaults.sparse_k,
            rerank_pool=defaults.rerank_pool,
            compression=defaults.compression,
        )
        return RetrievalPlan(
            steps=[step],
            target_tokens=defaults.target_tokens,
            coverage_target=defaults.coverage_target,
        )
