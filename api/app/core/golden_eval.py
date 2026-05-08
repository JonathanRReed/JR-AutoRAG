"""Golden set evaluation system for measuring RAG quality.

This module provides:
- GoldenTestCase: Expected question/answer/source pairs
- GoldenSetStore: Persistent storage for test datasets
- EvalMetrics: Retrieval and answer quality metrics
- GoldenSetEvaluator: Batch evaluation runner with regression detection
"""

from __future__ import annotations

import json
import hashlib
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config_snapshot import get_tool_versions

if TYPE_CHECKING:
    from .orchestrator import Orchestrator


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class GoldenTestCase:
    """A single test case with expected results."""
    question: str
    expected_source_ids: list[str] = field(default_factory=list)
    expected_answer_points: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_source_ids": self.expected_source_ids,
            "expected_answer_points": self.expected_answer_points,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenTestCase:
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            question=data["question"],
            expected_source_ids=data.get("expected_source_ids", []),
            expected_answer_points=data.get("expected_answer_points", []),
            tags=data.get("tags", []),
        )


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    recall_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    citation_coverage: float = 0.0  # % of answer claims backed by retrieved spans

    def to_dict(self) -> dict[str, float]:
        return {
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "citation_coverage": self.citation_coverage,
        }


@dataclass
class AnswerMetrics:
    """Answer quality metrics."""
    faithfulness: float = 0.0  # Grounded in sources
    completeness: float = 0.0  # Covers expected points
    refusal_accuracy: float = 0.0  # Correctly refused when no info
    coherence: float = 0.0  # Well-structured

    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "completeness": self.completeness,
            "refusal_accuracy": self.refusal_accuracy,
            "coherence": self.coherence,
        }


@dataclass
class TestCaseResult:
    """Result of evaluating a single test case."""
    test_case_id: str
    question: str
    answer: str
    retrieved_source_ids: list[str]
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics
    duration_ms: float = 0.0
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "question": self.question,
            "answer": self.answer,
            "retrieved_source_ids": self.retrieved_source_ids,
            "retrieval_metrics": self.retrieval_metrics.to_dict(),
            "answer_metrics": self.answer_metrics.to_dict(),
            "duration_ms": self.duration_ms,
            "trace_id": self.trace_id,
        }


@dataclass
class EvalRunResult:
    """Result of a complete evaluation run."""
    run_id: str
    golden_set_name: str
    timestamp: datetime
    retrieval_metrics: RetrievalMetrics  # Aggregated
    answer_metrics: AnswerMetrics  # Aggregated
    individual_results: list[TestCaseResult] = field(default_factory=list)
    duration_ms: float = 0.0
    audit: dict[str, Any] = field(default_factory=dict)
    report_path: str = ""
    report_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "golden_set_name": self.golden_set_name,
            "timestamp": self.timestamp.isoformat(),
            "retrieval_metrics": self.retrieval_metrics.to_dict(),
            "answer_metrics": self.answer_metrics.to_dict(),
            "individual_results": [r.to_dict() for r in self.individual_results],
            "duration_ms": self.duration_ms,
            "audit": self.audit,
            "report_path": self.report_path,
            "report_sha256": self.report_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRunResult:
        return cls(
            run_id=data["run_id"],
            golden_set_name=data["golden_set_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            retrieval_metrics=RetrievalMetrics(**data["retrieval_metrics"]),
            answer_metrics=AnswerMetrics(**data["answer_metrics"]),
            individual_results=[],  # Don't load full results for listing
            duration_ms=data.get("duration_ms", 0.0),
            audit=data.get("audit", {}),
            report_path=data.get("report_path", ""),
            report_sha256=data.get("report_sha256", ""),
        )


# ============================================================================
# Golden Set Store
# ============================================================================

class GoldenSetStore:
    """Persistent storage for golden test sets."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            path = Path("data/golden_sets.json")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._sets: dict[str, list[GoldenTestCase]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                for name, cases in data.items():
                    self._sets[name] = [
                        GoldenTestCase.from_dict(c) for c in cases
                    ]
            except (json.JSONDecodeError, KeyError):
                self._sets = {}

    def _save(self) -> None:
        data = {
            name: [c.to_dict() for c in cases]
            for name, cases in self._sets.items()
        }
        self._path.write_text(json.dumps(data, indent=2))

    def create_set(self, name: str, cases: list[GoldenTestCase]) -> None:
        """Create or replace a golden set."""
        self._sets[name] = cases
        self._save()

    def add_case(self, set_name: str, case: GoldenTestCase) -> None:
        """Add a test case to an existing set."""
        if set_name not in self._sets:
            self._sets[set_name] = []
        self._sets[set_name].append(case)
        self._save()

    def get_set(self, name: str) -> list[GoldenTestCase]:
        """Get a golden set by name."""
        return self._sets.get(name, [])

    def list_sets(self) -> list[dict[str, Any]]:
        """List all golden sets with metadata."""
        return [
            {"name": name, "count": len(cases)}
            for name, cases in self._sets.items()
        ]

    def delete_set(self, name: str) -> bool:
        """Delete a golden set."""
        if name in self._sets:
            del self._sets[name]
            self._save()
            return True
        return False


# ============================================================================
# Eval Run Store
# ============================================================================

class EvalRunStore:
    """Persistent storage for evaluation runs."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            path = Path("data/eval_runs.json")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._report_dir = self._path.parent / "eval_reports"
        self._runs: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._runs = json.loads(self._path.read_text())
            except json.JSONDecodeError:
                self._runs = {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._runs, indent=2))

    def save_run(self, result: EvalRunResult) -> None:
        """Save an evaluation run."""
        self._write_report(result)
        self._runs[result.run_id] = result.to_dict()
        self._save()

    def _write_report(self, result: EvalRunResult) -> None:
        """Write a standalone report artifact and attach its digest."""
        self._report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self._report_dir / f"{result.run_id}.json"
        result.report_path = str(report_path)
        result.report_sha256 = ""
        unsigned_payload = result.to_dict()
        encoded = json.dumps(unsigned_payload, indent=2, sort_keys=True, default=str).encode("utf-8")
        result.report_sha256 = hashlib.sha256(encoded).hexdigest()
        report_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def get_run(self, run_id: str) -> EvalRunResult | None:
        """Get a run by ID."""
        if run_id in self._runs:
            return EvalRunResult.from_dict(self._runs[run_id])
        return None

    def get_report(self, run_id: str) -> dict[str, Any] | None:
        """Load the durable report artifact for a run."""
        data = self._runs.get(run_id)
        if data is None:
            return None

        report_path = Path(data.get("report_path") or self._report_dir / f"{run_id}.json")
        try:
            resolved_report = report_path.resolve()
            resolved_report.relative_to(self._report_dir.resolve())
        except (OSError, ValueError):
            return None

        if not resolved_report.exists():
            return None
        try:
            return json.loads(resolved_report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent runs (summary only)."""
        runs = sorted(
            self._runs.values(),
            key=lambda r: r["timestamp"],
            reverse=True
        )[:limit]
        return [
            {
                "run_id": r["run_id"],
                "golden_set_name": r["golden_set_name"],
                "timestamp": r["timestamp"],
                "retrieval_metrics": r["retrieval_metrics"],
                "answer_metrics": r["answer_metrics"],
                "duration_ms": r.get("duration_ms", 0),
                "report_path": r.get("report_path", ""),
                "report_sha256": r.get("report_sha256", ""),
                "audit": r.get("audit", {}),
            }
            for r in runs
        ]


# ============================================================================
# Metrics Calculation
# ============================================================================

def compute_recall_at_k(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> float:
    """Compute recall@k: fraction of expected sources retrieved."""
    if not expected_ids:
        return 1.0  # No expected sources = perfect recall
    hits = sum(1 for eid in expected_ids if eid in retrieved_ids)
    return hits / len(expected_ids)


def compute_mrr(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> float:
    """Compute Mean Reciprocal Rank."""
    if not expected_ids:
        return 1.0
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in expected_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg(
    retrieved_ids: list[str],
    expected_ids: list[str],
) -> float:
    """Compute Normalized Discounted Cumulative Gain."""
    if not expected_ids:
        return 1.0

    # DCG: sum of relevance / log2(rank+1)
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid in expected_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all relevant in top positions
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(expected_ids)))

    return dcg / idcg if idcg > 0 else 0.0


def compute_completeness(
    answer: str,
    expected_points: list[str],
) -> float:
    """Compute answer completeness: fraction of expected points mentioned."""
    if not expected_points:
        return 1.0

    answer_lower = answer.lower()
    hits = sum(
        1 for point in expected_points
        if point.lower() in answer_lower
    )
    return hits / len(expected_points)


def compute_citation_coverage(
    answer: str,
    num_sources: int,
) -> float:
    """Estimate citation coverage from bracket citations in answer."""
    import re
    citations = re.findall(r'\[(\d+)\]', answer)
    if num_sources == 0:
        return 1.0 if not citations else 0.0
    unique_citations = len({int(c) for c in citations if c.isdigit()})
    return min(unique_citations / num_sources, 1.0)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================================
# Golden Set Evaluator
# ============================================================================

class GoldenSetEvaluator:
    """Runs batch evaluations against golden test sets."""

    def __init__(
        self,
        golden_store: GoldenSetStore | None = None,
        run_store: EvalRunStore | None = None,
    ) -> None:
        self.golden_store = golden_store or GoldenSetStore()
        self.run_store = run_store or EvalRunStore()

    async def run_batch(
        self,
        orchestrator: Orchestrator,
        set_name: str,
        on_progress: callable | None = None,
    ) -> EvalRunResult:
        """Run evaluation on a golden set."""
        test_cases = self.golden_store.get_set(set_name)
        if not test_cases:
            raise ValueError(f"Golden set '{set_name}' not found or empty")

        run_id = str(uuid.uuid4())[:12]
        start_time = time.perf_counter()
        individual_results: list[TestCaseResult] = []

        for i, case in enumerate(test_cases):
            if on_progress:
                on_progress(i + 1, len(test_cases), case.question)

            case_start = time.perf_counter()

            # Run the query
            response = await orchestrator.answer(case.question)

            case_duration = (time.perf_counter() - case_start) * 1000

            # Extract retrieved source IDs
            retrieved_ids = [
                s.get("id", "") for s in response.get("sources", [])
            ]

            # Compute retrieval metrics
            retrieval_metrics = RetrievalMetrics(
                recall_at_k=compute_recall_at_k(retrieved_ids, case.expected_source_ids),
                mrr=compute_mrr(retrieved_ids, case.expected_source_ids),
                ndcg=compute_ndcg(retrieved_ids, case.expected_source_ids),
                citation_coverage=compute_citation_coverage(
                    response.get("answer", ""),
                    len(response.get("sources", [])),
                ),
            )

            # Compute answer metrics
            answer_metrics = AnswerMetrics(
                completeness=compute_completeness(
                    response.get("answer", ""),
                    case.expected_answer_points,
                ),
                # Faithfulness requires LLM judge - use existing evaluator
                faithfulness=response.get("metrics", {}).get("faithfulness", 0.0),
                coherence=response.get("metrics", {}).get("coherence", 0.0),
            )

            individual_results.append(TestCaseResult(
                test_case_id=case.id,
                question=case.question,
                answer=response.get("answer", ""),
                retrieved_source_ids=retrieved_ids,
                retrieval_metrics=retrieval_metrics,
                answer_metrics=answer_metrics,
                duration_ms=case_duration,
                trace_id=response.get("trace_id", ""),
            ))

        # Aggregate metrics
        n = len(individual_results)
        agg_retrieval = RetrievalMetrics(
            recall_at_k=sum(r.retrieval_metrics.recall_at_k for r in individual_results) / n,
            mrr=sum(r.retrieval_metrics.mrr for r in individual_results) / n,
            ndcg=sum(r.retrieval_metrics.ndcg for r in individual_results) / n,
            citation_coverage=sum(r.retrieval_metrics.citation_coverage for r in individual_results) / n,
        )
        agg_answer = AnswerMetrics(
            faithfulness=sum(r.answer_metrics.faithfulness for r in individual_results) / n,
            completeness=sum(r.answer_metrics.completeness for r in individual_results) / n,
            coherence=sum(r.answer_metrics.coherence for r in individual_results) / n,
        )

        total_duration = (time.perf_counter() - start_time) * 1000
        audit = self._build_audit_snapshot(
            orchestrator=orchestrator,
            set_name=set_name,
            test_cases=test_cases,
            run_id=run_id,
            duration_ms=total_duration,
        )

        result = EvalRunResult(
            run_id=run_id,
            golden_set_name=set_name,
            timestamp=datetime.now(UTC),
            retrieval_metrics=agg_retrieval,
            answer_metrics=agg_answer,
            individual_results=individual_results,
            duration_ms=total_duration,
            audit=audit,
        )

        # Persist the run
        self.run_store.save_run(result)

        return result

    def _build_audit_snapshot(
        self,
        orchestrator: Orchestrator,
        set_name: str,
        test_cases: list[GoldenTestCase],
        run_id: str,
        duration_ms: float,
    ) -> dict[str, Any]:
        """Build a deterministic audit snapshot for a golden eval run."""
        context: dict[str, Any] = {}
        if hasattr(orchestrator, "get_eval_audit_context"):
            try:
                context = orchestrator.get_eval_audit_context()
            except Exception as exc:
                context = {"audit_context_error": str(exc)}

        cases_payload = [case.to_dict() for case in test_cases]
        return {
            "schema_version": "eval_run_audit_v1",
            "created_at": datetime.now(UTC).isoformat(),
            "run": {
                "run_id": run_id,
                "duration_ms": duration_ms,
            },
            "golden_set": {
                "name": set_name,
                "case_count": len(test_cases),
                "case_ids": [case.id for case in test_cases],
                "fingerprint": _canonical_sha256(cases_payload),
            },
            "corpus": context.get("corpus", {}),
            "runtime_profile": context.get("runtime_profile", {}),
            "model_status": context.get("model_status", {}),
            "config_snapshot": context.get("config_snapshot", {}),
            "tool_versions": get_tool_versions(),
            "context_error": context.get("audit_context_error", ""),
        }

    def compare_runs(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> dict[str, Any]:
        """Compare two evaluation runs to detect regressions."""
        run_a = self.run_store.get_run(run_id_a)
        run_b = self.run_store.get_run(run_id_b)

        if not run_a or not run_b:
            raise ValueError("One or both runs not found")

        def diff(a: float, b: float) -> dict[str, float]:
            return {
                "before": a,
                "after": b,
                "delta": b - a,
                "delta_pct": ((b - a) / a * 100) if a > 0 else 0,
            }

        return {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "retrieval": {
                "recall_at_k": diff(
                    run_a.retrieval_metrics.recall_at_k,
                    run_b.retrieval_metrics.recall_at_k,
                ),
                "mrr": diff(
                    run_a.retrieval_metrics.mrr,
                    run_b.retrieval_metrics.mrr,
                ),
                "ndcg": diff(
                    run_a.retrieval_metrics.ndcg,
                    run_b.retrieval_metrics.ndcg,
                ),
                "citation_coverage": diff(
                    run_a.retrieval_metrics.citation_coverage,
                    run_b.retrieval_metrics.citation_coverage,
                ),
            },
            "answer": {
                "faithfulness": diff(
                    run_a.answer_metrics.faithfulness,
                    run_b.answer_metrics.faithfulness,
                ),
                "completeness": diff(
                    run_a.answer_metrics.completeness,
                    run_b.answer_metrics.completeness,
                ),
            },
            "regressions": self._detect_regressions(run_a, run_b),
        }

    def _detect_regressions(
        self,
        before: EvalRunResult,
        after: EvalRunResult,
        threshold: float = 0.05,  # 5% regression threshold
    ) -> list[str]:
        """Detect significant regressions."""
        regressions = []

        checks = [
            ("recall_at_k", before.retrieval_metrics.recall_at_k, after.retrieval_metrics.recall_at_k),
            ("mrr", before.retrieval_metrics.mrr, after.retrieval_metrics.mrr),
            ("faithfulness", before.answer_metrics.faithfulness, after.answer_metrics.faithfulness),
            ("completeness", before.answer_metrics.completeness, after.answer_metrics.completeness),
        ]

        for name, before_val, after_val in checks:
            if before_val > 0 and (before_val - after_val) / before_val > threshold:
                regressions.append(
                    f"{name} dropped {((before_val - after_val) / before_val * 100):.1f}%"
                )

        return regressions


__all__ = [
    "GoldenTestCase",
    "RetrievalMetrics",
    "AnswerMetrics",
    "TestCaseResult",
    "EvalRunResult",
    "GoldenSetStore",
    "EvalRunStore",
    "GoldenSetEvaluator",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg",
    "compute_completeness",
    "compute_citation_coverage",
    "_canonical_sha256",
]
