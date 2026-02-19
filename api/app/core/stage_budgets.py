"""Stage budgets and timeouts for pipeline execution.

This module implements P0.2: Fast path and hard budgets.
- Per-stage timeout configuration
- Per-stage token budgets
- Graceful degradation when budgets exceeded
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger("autorag.budgets")

T = TypeVar("T")


@dataclass
class StageBudgetConfig:
    """Configuration for per-stage timeouts and token budgets.

    All timeout values are in milliseconds.
    Token budgets are approximate limits that trigger degradation.
    """

    # Timeouts per stage (milliseconds)
    planner_timeout_ms: int = 3000      # 3 seconds
    gatherer_timeout_ms: int = 12000    # 12 seconds
    rerank_timeout_ms: int = 5000       # 5 seconds
    compression_timeout_ms: int = 4000  # 4 seconds
    generation_timeout_ms: int = 20000  # 20 seconds
    verification_timeout_ms: int = 5000 # 5 seconds

    # Token budgets per stage
    retrieval_token_budget: int = 8000   # Max tokens from retrieval
    rerank_pool_budget: int = 50         # Max candidates to rerank
    compression_token_budget: int = 4000 # Target after compression
    answer_token_budget: int = 2000      # Max answer length

    # Overall pipeline budget
    total_timeout_ms: int = 60000        # 60 second hard cap

    def to_dict(self) -> dict[str, int]:
        return {
            "planner_timeout_ms": self.planner_timeout_ms,
            "gatherer_timeout_ms": self.gatherer_timeout_ms,
            "rerank_timeout_ms": self.rerank_timeout_ms,
            "compression_timeout_ms": self.compression_timeout_ms,
            "generation_timeout_ms": self.generation_timeout_ms,
            "verification_timeout_ms": self.verification_timeout_ms,
            "retrieval_token_budget": self.retrieval_token_budget,
            "rerank_pool_budget": self.rerank_pool_budget,
            "compression_token_budget": self.compression_token_budget,
            "answer_token_budget": self.answer_token_budget,
            "total_timeout_ms": self.total_timeout_ms,
        }


@dataclass
class StageResult:
    """Result from a stage execution with budget tracking."""

    value: Any
    stage: str
    duration_ms: float
    timed_out: bool = False
    used_fallback: bool = False
    tokens_used: int | None = None
    budget_exceeded: bool = False
    degradation_reason: str | None = None


@dataclass
class DegradationPath:
    """Defines how to degrade when a stage times out or exceeds budget."""

    stage: str
    fallback_fn: Callable[..., Any] | None = None
    reduced_params: dict[str, Any] = field(default_factory=dict)
    skip_allowed: bool = False


# Default degradation paths for each stage
DEFAULT_DEGRADATIONS: dict[str, DegradationPath] = {
    "planner": DegradationPath(
        stage="planner",
        reduced_params={"use_heuristic": True},  # Fall back to heuristic planner
        skip_allowed=False,
    ),
    "gatherer": DegradationPath(
        stage="gatherer",
        reduced_params={"top_k": 5, "skip_parallel": True},  # Reduce retrieval scope
        skip_allowed=False,
    ),
    "rerank": DegradationPath(
        stage="rerank",
        skip_allowed=True,  # Can skip reranking entirely
    ),
    "compression": DegradationPath(
        stage="compression",
        skip_allowed=True,  # Can use uncompressed context
    ),
    "generation": DegradationPath(
        stage="generation",
        reduced_params={"max_tokens": 500},  # Shorter answer
        skip_allowed=False,
    ),
    "verification": DegradationPath(
        stage="verification",
        skip_allowed=True,  # Can skip verification in fast path
    ),
    "graph_build": DegradationPath(
        stage="graph_build",
        skip_allowed=True,  # Graph is optional
    ),
    "graph_retrieval": DegradationPath(
        stage="graph_retrieval",
        skip_allowed=True,  # Graph retrieval is optional
    ),
}


class StageBudgetEnforcer:
    """Enforces time and token budgets with graceful degradation.

    Wraps async stage execution with timeout handling and provides
    fallback execution when budgets are exceeded.
    """

    def __init__(
        self,
        config: StageBudgetConfig | None = None,
        degradations: dict[str, DegradationPath] | None = None,
    ) -> None:
        self._config = config or StageBudgetConfig()
        self._degradations = degradations or DEFAULT_DEGRADATIONS
        self._stage_stats: dict[str, list[float]] = {}  # Track timing history

    def get_timeout(self, stage: str) -> float:
        """Get timeout in seconds for a stage."""
        timeout_map = {
            "planner": self._config.planner_timeout_ms,
            "planning": self._config.planner_timeout_ms,
            "gatherer": self._config.gatherer_timeout_ms,
            "gathering": self._config.gatherer_timeout_ms,
            "rerank": self._config.rerank_timeout_ms,
            "reranking": self._config.rerank_timeout_ms,
            "compression": self._config.compression_timeout_ms,
            "generation": self._config.generation_timeout_ms,
            "generating": self._config.generation_timeout_ms,
            "verification": self._config.verification_timeout_ms,
            "graph_build": self._config.gatherer_timeout_ms,  # Use gatherer timeout
            "graph_retrieval": self._config.rerank_timeout_ms,
        }
        timeout_ms = timeout_map.get(stage, 10000)  # Default 10s
        return timeout_ms / 1000.0

    def get_token_budget(self, stage: str) -> int | None:
        """Get token budget for a stage, if applicable."""
        budget_map = {
            "retrieval": self._config.retrieval_token_budget,
            "gatherer": self._config.retrieval_token_budget,
            "rerank": self._config.rerank_pool_budget,
            "compression": self._config.compression_token_budget,
            "generation": self._config.answer_token_budget,
        }
        return budget_map.get(stage)

    async def run_with_budget(
        self,
        stage: str,
        coro: Coroutine[Any, Any, T],
        fallback: T | Callable[[], T] | None = None,
        fallback_coro: Coroutine[Any, Any, T] | None = None,
    ) -> StageResult:
        """Run a stage with timeout enforcement and fallback.

        Args:
            stage: Stage name (e.g., "planner", "gatherer")
            coro: Async coroutine to execute
            fallback: Sync fallback value or callable if timeout
            fallback_coro: Async fallback coroutine if timeout

        Returns:
            StageResult with value and budget tracking info
        """
        import time

        timeout = self.get_timeout(stage)
        start = time.perf_counter()

        try:
            value = await asyncio.wait_for(coro, timeout=timeout)
            duration_ms = (time.perf_counter() - start) * 1000

            # Track timing for estimation
            if stage not in self._stage_stats:
                self._stage_stats[stage] = []
            self._stage_stats[stage].append(duration_ms)
            # Keep last 20 measurements
            self._stage_stats[stage] = self._stage_stats[stage][-20:]

            return StageResult(
                value=value,
                stage=stage,
                duration_ms=duration_ms,
                timed_out=False,
                used_fallback=False,
            )

        except TimeoutError:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "Stage %s timed out after %.1fms (limit: %.1fms)",
                stage, duration_ms, timeout * 1000
            )

            # Check degradation path
            degradation = self._degradations.get(stage)

            # Try fallback
            if fallback_coro is not None:
                try:
                    # Give fallback half the original timeout
                    value = await asyncio.wait_for(
                        fallback_coro,
                        timeout=timeout / 2
                    )
                    return StageResult(
                        value=value,
                        stage=stage,
                        duration_ms=(time.perf_counter() - start) * 1000,
                        timed_out=True,
                        used_fallback=True,
                        degradation_reason="timeout_async_fallback",
                    )
                except Exception:
                    pass

            if fallback is not None:
                value = fallback() if callable(fallback) else fallback
                return StageResult(
                    value=value,
                    stage=stage,
                    duration_ms=duration_ms,
                    timed_out=True,
                    used_fallback=True,
                    degradation_reason="timeout_sync_fallback",
                )

            # Check if stage can be skipped
            if degradation and degradation.skip_allowed:
                return StageResult(
                    value=None,
                    stage=stage,
                    duration_ms=duration_ms,
                    timed_out=True,
                    used_fallback=True,
                    degradation_reason="timeout_skipped",
                )

            # Re-raise if no fallback available
            raise

    def check_token_budget(
        self,
        stage: str,
        tokens: int,
    ) -> tuple[bool, int | None]:
        """Check if token count exceeds budget.

        Returns:
            Tuple of (exceeded: bool, budget: int | None)
        """
        budget = self.get_token_budget(stage)
        if budget is None:
            return False, None
        return tokens > budget, budget

    def enforce_token_budget(
        self,
        stage: str,
        items: list[T],
        token_counter: Callable[[T], int],
    ) -> tuple[list[T], int]:
        """Truncate items to fit within token budget.

        Args:
            stage: Stage name
            items: List of items (chunks, candidates, etc.)
            token_counter: Function to count tokens per item

        Returns:
            Tuple of (truncated items, tokens dropped)
        """
        budget = self.get_token_budget(stage)
        if budget is None:
            return items, 0

        kept = []
        total_tokens = 0
        dropped_tokens = 0

        for item in items:
            item_tokens = token_counter(item)
            if total_tokens + item_tokens <= budget:
                kept.append(item)
                total_tokens += item_tokens
            else:
                dropped_tokens += item_tokens

        if dropped_tokens > 0:
            logger.info(
                "Stage %s: kept %d items (%d tokens), dropped %d tokens",
                stage, len(kept), total_tokens, dropped_tokens
            )

        return kept, dropped_tokens

    def get_estimated_latency(self, stage: str) -> tuple[float, float] | None:
        """Get estimated latency range from recent history.

        Returns:
            Tuple of (p25, p75) latency in ms, or None if no data
        """
        stats = self._stage_stats.get(stage)
        if not stats or len(stats) < 3:
            return None

        sorted_stats = sorted(stats)
        p25_idx = len(sorted_stats) // 4
        p75_idx = (len(sorted_stats) * 3) // 4

        return sorted_stats[p25_idx], sorted_stats[p75_idx]

    def get_all_estimates(self) -> dict[str, tuple[float, float] | None]:
        """Get estimated latency for all tracked stages."""
        return {
            stage: self.get_estimated_latency(stage)
            for stage in self._stage_stats
        }


# Global instance
_budget_enforcer: StageBudgetEnforcer | None = None


def get_budget_enforcer(config: StageBudgetConfig | None = None) -> StageBudgetEnforcer:
    """Get or create global budget enforcer."""
    global _budget_enforcer
    if _budget_enforcer is None or config is not None:
        _budget_enforcer = StageBudgetEnforcer(config)
    return _budget_enforcer


__all__ = [
    "StageBudgetConfig",
    "StageResult",
    "DegradationPath",
    "StageBudgetEnforcer",
    "get_budget_enforcer",
    "DEFAULT_DEGRADATIONS",
]
