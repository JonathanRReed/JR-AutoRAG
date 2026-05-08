"""Health check endpoints for Kubernetes-style probes.

/healthz - Liveness probe: Is the process running?
/readyz  - Readiness probe: Is the service ready to accept traffic?
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from ..schemas.health import ReadinessResponse
from ..state import get_orchestrator

logger = logging.getLogger("autorag.health")

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe - is the process alive?"""
    return {"status": "ok"}


@router.get("/readyz", response_model=ReadinessResponse)
def readyz() -> JSONResponse:
    """Readiness probe - is the service ready to accept traffic?

    Checks:
    - Orchestrator initialized
    - Document store accessible
    - Vector index loaded
    - Embedding model available
    - At least one LLM provider configured

    Returns 200 if all checks pass, 503 if any critical check fails.
    """
    try:
        orchestrator = get_orchestrator()
        if orchestrator is None:
            response = ReadinessResponse(
                ready=False,
                level="not_ready",
                checks={
                    "orchestrator": {
                        "status": "fail",
                        "message": "Orchestrator not initialized",
                        "details": {},
                    }
                },
            )
        else:
            response = ReadinessResponse.model_validate(orchestrator.get_readiness_status())
    except Exception as exc:
        response = ReadinessResponse(
            ready=False,
            level="not_ready",
            checks={
                "orchestrator": {
                    "status": "fail",
                    "message": f"Error building readiness report: {exc}",
                    "details": {},
                }
            },
        )

    status_code = status.HTTP_200_OK if response.ready else status.HTTP_503_SERVICE_UNAVAILABLE
    if response.ready:
        logger.debug("Readiness check passed: %s", response.checks)
    else:
        logger.warning("Readiness check failed: %s", response.checks)
    return JSONResponse(content=response.model_dump(mode="json"), status_code=status_code)
