"""Evaluation endpoints (simple sequential runner)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..schemas.evaluation import EvaluationRequest, EvaluationRun
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("", response_model=EvaluationRun)
async def run_evaluation(payload: EvaluationRequest, container: ServiceContainer = Depends(get_container)):
    if not payload.questions:
        raise HTTPException(status_code=400, detail="Must supply at least one question")

    import asyncio

    if not container.document_store.list():
        responses = []
        for question in payload.questions:
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
