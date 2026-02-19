"""Trace replay and diff for reproducibility and debugging.

Provides:
- TraceReplayer: Re-run queries using exact config snapshots
- TraceDiff: Compare two runs to identify differences
- ReplayResult: Detailed comparison of original vs replayed results

Implements the "Replay this run" and "Trace diff" features for
debugging and ensuring reproducibility.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .orchestrator import Orchestrator


# =============================================================================
# Delta Types
# =============================================================================

@dataclass
class RetrievalDelta:
    """Differences in retrieval results between two runs."""
    added_docs: list[str]  # Doc IDs in new but not old
    removed_docs: list[str]  # Doc IDs in old but not new
    score_changes: dict[str, tuple[float, float]]  # doc_id -> (old_score, new_score)
    position_changes: dict[str, tuple[int, int]]  # doc_id -> (old_pos, new_pos)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_docs": self.added_docs,
            "removed_docs": self.removed_docs,
            "score_changes": self.score_changes,
            "position_changes": self.position_changes,
            "summary": self.summary(),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        parts = []
        if self.added_docs:
            parts.append(f"+{len(self.added_docs)} docs")
        if self.removed_docs:
            parts.append(f"-{len(self.removed_docs)} docs")
        if self.position_changes:
            parts.append(f"{len(self.position_changes)} position changes")
        return ", ".join(parts) if parts else "No changes"

    @property
    def has_changes(self) -> bool:
        return bool(self.added_docs or self.removed_docs or self.position_changes)


@dataclass
class RerankerDelta:
    """Differences in reranking results between two runs."""
    score_changes: dict[str, tuple[float, float]]  # doc_id -> (old_score, new_score)
    order_changed: bool
    new_order: list[str]
    old_order: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_changes": self.score_changes,
            "order_changed": self.order_changed,
            "new_order": self.new_order,
            "old_order": self.old_order,
        }

    @property
    def has_changes(self) -> bool:
        return self.order_changed or bool(self.score_changes)


@dataclass
class PromptDelta:
    """Differences in prompts between two runs."""
    system_prompt_diff: list[str]  # Unified diff lines
    query_template_diff: list[str]
    final_prompt_diff: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt_diff": self.system_prompt_diff,
            "query_template_diff": self.query_template_diff,
            "final_prompt_diff": self.final_prompt_diff,
        }

    @property
    def has_changes(self) -> bool:
        return bool(
            self.system_prompt_diff or
            self.query_template_diff or
            self.final_prompt_diff
        )


@dataclass
class LatencyDelta:
    """Differences in latency between two runs."""
    total_ms: tuple[float, float]  # (old, new)
    per_stage: dict[str, tuple[float, float]]  # stage -> (old_ms, new_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ms": self.total_ms,
            "per_stage": self.per_stage,
            "total_change_pct": self._calc_change_pct(self.total_ms),
        }

    def _calc_change_pct(self, values: tuple[float, float]) -> float:
        old, new = values
        if old == 0:
            return 0.0
        return ((new - old) / old) * 100


@dataclass
class OutputDelta:
    """Differences in output between two runs."""
    answer_diff: list[str]  # Unified diff lines
    citations_added: list[str]
    citations_removed: list[str]
    confidence_change: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_diff": self.answer_diff,
            "citations_added": self.citations_added,
            "citations_removed": self.citations_removed,
            "confidence_change": self.confidence_change,
            "answers_identical": len(self.answer_diff) == 0,
        }

    @property
    def has_changes(self) -> bool:
        return bool(
            self.answer_diff or
            self.citations_added or
            self.citations_removed
        )


# =============================================================================
# Trace Diff
# =============================================================================

@dataclass
class TraceDiff:
    """Complete diff between two trace bundles."""
    trace_a_id: str
    trace_b_id: str
    timestamp: str

    config_identical: bool
    corpus_identical: bool

    retrieval: RetrievalDelta
    reranker: RerankerDelta
    prompts: PromptDelta
    latency: LatencyDelta
    output: OutputDelta

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_a_id": self.trace_a_id,
            "trace_b_id": self.trace_b_id,
            "timestamp": self.timestamp,
            "config_identical": self.config_identical,
            "corpus_identical": self.corpus_identical,
            "retrieval": self.retrieval.to_dict(),
            "reranker": self.reranker.to_dict(),
            "prompts": self.prompts.to_dict(),
            "latency": self.latency.to_dict(),
            "output": self.output.to_dict(),
            "summary": self.summary(),
        }

    def summary(self) -> str:
        """Generate human-readable summary of differences."""
        changes = []

        if not self.config_identical:
            changes.append("Config differs")
        if not self.corpus_identical:
            changes.append("Corpus differs")
        if self.retrieval.has_changes:
            changes.append(f"Retrieval: {self.retrieval.summary()}")
        if self.reranker.has_changes:
            changes.append("Reranking order changed")
        if self.prompts.has_changes:
            changes.append("Prompts differ")
        if self.output.has_changes:
            changes.append("Outputs differ")

        if not changes:
            return "Runs are identical"

        return "; ".join(changes)

    @property
    def is_identical(self) -> bool:
        """Check if the two runs are functionally identical."""
        return (
            self.config_identical and
            self.corpus_identical and
            not self.retrieval.has_changes and
            not self.reranker.has_changes and
            not self.output.has_changes
        )


class TraceDiffer:
    """Compare two trace bundles to identify differences."""

    def diff(
        self,
        trace_a: dict[str, Any],
        trace_b: dict[str, Any],
    ) -> TraceDiff:
        """Compare two trace bundles.

        Args:
            trace_a: First trace bundle (baseline)
            trace_b: Second trace bundle (comparison)

        Returns:
            TraceDiff with detailed comparison
        """
        # Extract IDs
        trace_a_id = trace_a.get("trace_id", "unknown")
        trace_b_id = trace_b.get("trace_id", "unknown")

        # Compare configs
        config_a = trace_a.get("config_snapshot", {})
        config_b = trace_b.get("config_snapshot", {})
        config_identical = (
            config_a.get("snapshot_id") == config_b.get("snapshot_id")
        )

        # Compare corpus
        corpus_a = config_a.get("corpus_hash", "")
        corpus_b = config_b.get("corpus_hash", "")
        corpus_identical = corpus_a == corpus_b

        # Compute deltas
        retrieval = self._diff_retrieval(
            trace_a.get("retrieval", {}),
            trace_b.get("retrieval", {}),
        )

        reranker = self._diff_reranker(
            trace_a.get("reranker", {}),
            trace_b.get("reranker", {}),
        )

        prompts = self._diff_prompts(
            trace_a.get("prompts", {}),
            trace_b.get("prompts", {}),
        )

        latency = self._diff_latency(
            trace_a.get("metrics", {}),
            trace_b.get("metrics", {}),
        )

        output = self._diff_output(
            trace_a.get("output", {}),
            trace_b.get("output", {}),
        )

        return TraceDiff(
            trace_a_id=trace_a_id,
            trace_b_id=trace_b_id,
            timestamp=datetime.utcnow().isoformat(),
            config_identical=config_identical,
            corpus_identical=corpus_identical,
            retrieval=retrieval,
            reranker=reranker,
            prompts=prompts,
            latency=latency,
            output=output,
        )

    def _diff_retrieval(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> RetrievalDelta:
        """Compare retrieval results."""
        docs_a = {d["id"]: d for d in a.get("documents", [])}
        docs_b = {d["id"]: d for d in b.get("documents", [])}

        ids_a = set(docs_a.keys())
        ids_b = set(docs_b.keys())

        added = list(ids_b - ids_a)
        removed = list(ids_a - ids_b)

        # Score changes for common docs
        score_changes = {}
        for doc_id in ids_a & ids_b:
            score_a = docs_a[doc_id].get("score", 0)
            score_b = docs_b[doc_id].get("score", 0)
            if abs(score_a - score_b) > 0.001:
                score_changes[doc_id] = (score_a, score_b)

        # Position changes
        order_a = list(docs_a.keys())
        order_b = list(docs_b.keys())
        position_changes = {}
        for doc_id in ids_a & ids_b:
            pos_a = order_a.index(doc_id) if doc_id in order_a else -1
            pos_b = order_b.index(doc_id) if doc_id in order_b else -1
            if pos_a != pos_b:
                position_changes[doc_id] = (pos_a, pos_b)

        return RetrievalDelta(
            added_docs=added,
            removed_docs=removed,
            score_changes=score_changes,
            position_changes=position_changes,
        )

    def _diff_reranker(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> RerankerDelta:
        """Compare reranker results."""
        scores_a = {d["id"]: d.get("score", 0) for d in a.get("documents", [])}
        scores_b = {d["id"]: d.get("score", 0) for d in b.get("documents", [])}

        order_a = [d["id"] for d in a.get("documents", [])]
        order_b = [d["id"] for d in b.get("documents", [])]

        score_changes = {}
        for doc_id in set(scores_a.keys()) & set(scores_b.keys()):
            if abs(scores_a[doc_id] - scores_b[doc_id]) > 0.001:
                score_changes[doc_id] = (scores_a[doc_id], scores_b[doc_id])

        return RerankerDelta(
            score_changes=score_changes,
            order_changed=order_a != order_b,
            old_order=order_a,
            new_order=order_b,
        )

    def _diff_prompts(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> PromptDelta:
        """Compare prompts."""
        system_a = a.get("system_prompt", "")
        system_b = b.get("system_prompt", "")

        query_a = a.get("query_template", "")
        query_b = b.get("query_template", "")

        final_a = a.get("final_prompt", "")
        final_b = b.get("final_prompt", "")

        return PromptDelta(
            system_prompt_diff=list(difflib.unified_diff(
                system_a.splitlines(),
                system_b.splitlines(),
                lineterm="",
            )),
            query_template_diff=list(difflib.unified_diff(
                query_a.splitlines(),
                query_b.splitlines(),
                lineterm="",
            )),
            final_prompt_diff=list(difflib.unified_diff(
                final_a.splitlines(),
                final_b.splitlines(),
                lineterm="",
            )),
        )

    def _diff_latency(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> LatencyDelta:
        """Compare latency metrics."""
        total_a = a.get("total_ms", 0)
        total_b = b.get("total_ms", 0)

        stages_a = a.get("per_stage", {})
        stages_b = b.get("per_stage", {})

        all_stages = set(stages_a.keys()) | set(stages_b.keys())
        per_stage = {}
        for stage in all_stages:
            per_stage[stage] = (
                stages_a.get(stage, 0),
                stages_b.get(stage, 0),
            )

        return LatencyDelta(
            total_ms=(total_a, total_b),
            per_stage=per_stage,
        )

    def _diff_output(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
    ) -> OutputDelta:
        """Compare outputs."""
        answer_a = a.get("answer", "")
        answer_b = b.get("answer", "")

        citations_a = set(a.get("citations", []))
        citations_b = set(b.get("citations", []))

        confidence_a = a.get("confidence", 0)
        confidence_b = b.get("confidence", 0)

        return OutputDelta(
            answer_diff=list(difflib.unified_diff(
                answer_a.splitlines(),
                answer_b.splitlines(),
                lineterm="",
            )),
            citations_added=list(citations_b - citations_a),
            citations_removed=list(citations_a - citations_b),
            confidence_change=(confidence_a, confidence_b),
        )


# =============================================================================
# Trace Replayer
# =============================================================================

@dataclass
class ReplayResult:
    """Result of replaying a trace."""
    original_trace_id: str
    replay_trace_id: str
    timestamp: str
    success: bool
    error: str | None
    diff: TraceDiff | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_trace_id": self.original_trace_id,
            "replay_trace_id": self.replay_trace_id,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
            "is_identical": self.diff.is_identical if self.diff else None,
            "diff": self.diff.to_dict() if self.diff else None,
        }


class TraceReplayer:
    """Replay traces using their original configuration.

    Enables "Replay this run" functionality for debugging
    and verifying reproducibility.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        self._orchestrator = orchestrator
        self._differ = TraceDiffer()

    async def replay(
        self,
        trace_bundle: dict[str, Any],
        compare: bool = True,
    ) -> ReplayResult:
        """Replay a query using the trace's original configuration.

        Args:
            trace_bundle: Original trace bundle to replay
            compare: Whether to compare results with original

        Returns:
            ReplayResult with comparison if requested
        """
        original_trace_id = trace_bundle.get("trace_id", "unknown")
        timestamp = datetime.utcnow().isoformat()

        try:
            # Extract original query and configuration
            query = trace_bundle.get("query", "")
            trace_bundle.get("config_snapshot", {})

            if not query:
                return ReplayResult(
                    original_trace_id=original_trace_id,
                    replay_trace_id="",
                    timestamp=timestamp,
                    success=False,
                    error="No query found in trace bundle",
                    diff=None,
                )

            # TODO: Apply config snapshot settings to orchestrator
            # For now, we replay with current settings
            # In full implementation, we would:
            # 1. Save current config
            # 2. Apply snapshot config
            # 3. Run query
            # 4. Restore original config

            # Run the query
            result = await self._orchestrator.answer(
                query=query,
                trace_id=f"replay_{original_trace_id}",
            )

            # Get the new trace
            new_trace = result.get("trace_bundle", {})
            replay_trace_id = new_trace.get("trace_id", "unknown")

            # Compare if requested
            diff = None
            if compare and new_trace:
                diff = self._differ.diff(trace_bundle, new_trace)

            return ReplayResult(
                original_trace_id=original_trace_id,
                replay_trace_id=replay_trace_id,
                timestamp=timestamp,
                success=True,
                error=None,
                diff=diff,
            )

        except Exception as e:
            return ReplayResult(
                original_trace_id=original_trace_id,
                replay_trace_id="",
                timestamp=timestamp,
                success=False,
                error=str(e),
                diff=None,
            )

    def diff(
        self,
        trace_a: dict[str, Any],
        trace_b: dict[str, Any],
    ) -> TraceDiff:
        """Compare two traces without replaying."""
        return self._differ.diff(trace_a, trace_b)


# =============================================================================
# Singleton differ
# =============================================================================

_trace_differ: TraceDiffer | None = None


def get_trace_differ() -> TraceDiffer:
    """Get the global trace differ instance."""
    global _trace_differ
    if _trace_differ is None:
        _trace_differ = TraceDiffer()
    return _trace_differ


__all__ = [
    "TraceDiff",
    "TraceDiffer",
    "TraceReplayer",
    "ReplayResult",
    "RetrievalDelta",
    "RerankerDelta",
    "PromptDelta",
    "LatencyDelta",
    "OutputDelta",
    "get_trace_differ",
]
