"""Evaluation endpoints with golden set evaluation system."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..core.golden_eval import (
    EvalRunStore,
    GoldenSetEvaluator,
    GoldenSetStore,
    GoldenTestCase,
)
from ..schemas.evaluation import (
    AnswerMetricsSchema,
    EvalRunResultSchema,
    EvalRunSummary,
    EvaluationRequest,
    EvaluationRun,
    GoldenSetCreateRequest,
    GoldenSetInfo,
    GoldenTestCaseSchema,
    RetrievalMetricsSchema,
    RunComparisonResult,
    TestCaseResultSchema,
)
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Initialize stores (singleton pattern via module-level)
_golden_store: GoldenSetStore | None = None
_eval_run_store: EvalRunStore | None = None
_evaluator: GoldenSetEvaluator | None = None


def get_golden_store() -> GoldenSetStore:
    global _golden_store
    if _golden_store is None:
        _golden_store = GoldenSetStore()
    return _golden_store


def get_eval_run_store() -> EvalRunStore:
    global _eval_run_store
    if _eval_run_store is None:
        _eval_run_store = EvalRunStore()
    return _eval_run_store


def get_evaluator() -> GoldenSetEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = GoldenSetEvaluator(
            golden_store=get_golden_store(),
            run_store=get_eval_run_store(),
        )
    return _evaluator


# ============================================================================
# Original Evaluation Endpoint (preserved)
# ============================================================================

@router.post("", response_model=EvaluationRun)
async def run_evaluation(payload: EvaluationRequest, container: ServiceContainer = Depends(get_container)):
    """Run ad-hoc evaluation on a list of questions."""
    if not payload.questions:
        raise HTTPException(status_code=400, detail="Must supply at least one question")

    import asyncio

    if not container.document_store.list():
        responses = []
        for _question in payload.questions:
            responses.append({
                "answer": "Demo mode: No documents ingested yet. Upload files to enable retrieval.",
                "chunks": [],
                "sources": [],
                "trace_id": "demo",
                "metrics": {
                    "chunks": 0,
                    "coverage": 0.0,
                    "tokens": 0,
                    "duration_ms": 0.0,
                    "query_type": "demo",
                },
                "steps": [],
            })
        return EvaluationRun(
            name=payload.name,
            responses=responses,
            average_coverage=0.0,
            average_tokens=0.0,
        )

    # Run all queries in parallel
    tasks = [container.orchestrator.answer(q) for q in payload.questions]
    responses = await asyncio.gather(*tasks)

    avg_coverage = sum(r["metrics"].get("coverage", 0.0) for r in responses) / len(responses)
    avg_tokens = sum(r["metrics"].get("tokens", 0.0) for r in responses) / len(responses)

    return EvaluationRun(
        name=payload.name,
        responses=responses,
        average_coverage=avg_coverage,
        average_tokens=avg_tokens,
    )


# ============================================================================
# Golden Set Management Endpoints
# ============================================================================

@router.post("/golden-sets", response_model=GoldenSetInfo)
async def create_golden_set(payload: GoldenSetCreateRequest):
    """Create or replace a golden test set."""
    store = get_golden_store()

    cases = [
        GoldenTestCase(
            question=c.question,
            expected_source_ids=c.expected_source_ids,
            expected_answer_points=c.expected_answer_points,
            tags=c.tags,
            id=c.id or "",
        )
        for c in payload.cases
    ]

    store.create_set(payload.name, cases)

    return GoldenSetInfo(name=payload.name, count=len(cases))


@router.get("/golden-sets", response_model=list[GoldenSetInfo])
async def list_golden_sets():
    """List all golden test sets."""
    store = get_golden_store()
    return [GoldenSetInfo(**info) for info in store.list_sets()]


@router.get("/golden-sets/{set_name}", response_model=list[GoldenTestCaseSchema])
async def get_golden_set(set_name: str):
    """Get a specific golden set."""
    store = get_golden_store()
    cases = store.get_set(set_name)
    if not cases:
        raise HTTPException(status_code=404, detail=f"Golden set '{set_name}' not found")

    return [
        GoldenTestCaseSchema(
            id=c.id,
            question=c.question,
            expected_source_ids=c.expected_source_ids,
            expected_answer_points=c.expected_answer_points,
            tags=c.tags,
        )
        for c in cases
    ]


@router.delete("/golden-sets/{set_name}")
async def delete_golden_set(set_name: str):
    """Delete a golden set."""
    store = get_golden_store()
    if not store.delete_set(set_name):
        raise HTTPException(status_code=404, detail=f"Golden set '{set_name}' not found")
    return {"deleted": set_name}


# ============================================================================
# Batch Evaluation Endpoints
# ============================================================================

@router.post("/batch/{set_name}", response_model=EvalRunResultSchema)
async def run_batch_evaluation(
    set_name: str,
    container: ServiceContainer = Depends(get_container),
):
    """Run batch evaluation on a golden set."""
    evaluator = get_evaluator()

    try:
        result = await evaluator.run_batch(
            orchestrator=container.orchestrator,
            set_name=set_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Convert to schema
    return EvalRunResultSchema(
        run_id=result.run_id,
        golden_set_name=result.golden_set_name,
        timestamp=result.timestamp,
        retrieval_metrics=RetrievalMetricsSchema(**result.retrieval_metrics.to_dict()),
        answer_metrics=AnswerMetricsSchema(**result.answer_metrics.to_dict()),
        individual_results=[
            TestCaseResultSchema(
                test_case_id=r.test_case_id,
                question=r.question,
                answer=r.answer,
                retrieved_source_ids=r.retrieved_source_ids,
                retrieval_metrics=RetrievalMetricsSchema(**r.retrieval_metrics.to_dict()),
                answer_metrics=AnswerMetricsSchema(**r.answer_metrics.to_dict()),
                duration_ms=r.duration_ms,
                trace_id=r.trace_id,
            )
            for r in result.individual_results
        ],
        duration_ms=result.duration_ms,
        audit=result.audit,
        report_path=result.report_path,
        report_sha256=result.report_sha256,
    )


@router.get("/runs", response_model=list[EvalRunSummary])
async def list_eval_runs(limit: int = 50):
    """List recent evaluation runs."""
    store = get_eval_run_store()
    runs = store.list_runs(limit=limit)
    return [EvalRunSummary(**run) for run in runs]


@router.get("/runs/{run_id}/report", response_model=dict[str, Any])
async def get_eval_run_report(run_id: str):
    """Get the durable JSON report artifact for an evaluation run."""
    store = get_eval_run_store()
    report = store.get_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Evaluation report '{run_id}' not found")
    return report


@router.get("/runs/{run_id}/compare/{other_run_id}", response_model=RunComparisonResult)
async def compare_eval_runs(run_id: str, other_run_id: str):
    """Compare two evaluation runs to detect regressions."""
    evaluator = get_evaluator()

    try:
        comparison = evaluator.compare_runs(run_id, other_run_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return RunComparisonResult(**comparison)
