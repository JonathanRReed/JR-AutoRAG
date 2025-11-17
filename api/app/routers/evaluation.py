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
    responses = []
    for question in payload.questions:
        result = await container.orchestrator.answer(question)
        responses.append(result)
    avg_coverage = sum(r["metrics"].get("coverage", 0.0) for r in responses) / len(responses)
    avg_tokens = sum(r["metrics"].get("tokens", 0.0) for r in responses) / len(responses)
    return EvaluationRun(
        name=payload.name,
        responses=responses,
        average_coverage=avg_coverage,
        average_tokens=avg_tokens,
    )
