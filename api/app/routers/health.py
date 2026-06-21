"""Health check endpoints for Kubernetes-style probes.

/healthz - Liveness probe: Is the process running?
/readyz  - Readiness probe: Is the service ready to accept traffic?
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from ..schemas.health import ReadinessCheck, ReadinessResponse
from ..state import get_orchestrator

logger = logging.getLogger("autorag.health")

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe - is the process alive?"""
    return {"status": "ok"}


def _public_readiness_response(response: ReadinessResponse) -> ReadinessResponse:
    """Strip operator-only diagnostics from the unauthenticated readiness response."""
    return ReadinessResponse(
        ready=response.ready,
        level=response.level,
        checks={
            name: ReadinessCheck(status=check.status)
            for name, check in response.checks.items()
        },
    )


@router.get("/readyz", response_model=ReadinessResponse)
def readyz() -> JSONResponse:
    """Readiness probe - is the service ready to accept traffic?

    Returns a minimal public contract suitable for unauthenticated Kubernetes
    and load-balancer probes. Detailed operator diagnostics are intentionally
    omitted from this endpoint.
    """
    try:
        orchestrator = get_orchestrator()
        if orchestrator is None:
            response = ReadinessResponse(
                ready=False,
                level="not_ready",
                checks={"orchestrator": ReadinessCheck(status="fail")},
            )
        else:
            internal_response = ReadinessResponse.model_validate(orchestrator.get_readiness_status())
            response = _public_readiness_response(internal_response)
    except Exception:
        logger.exception("Failed to build readiness report")
        response = ReadinessResponse(
            ready=False,
            level="not_ready",
            checks={"orchestrator": ReadinessCheck(status="fail")},
        )

    status_code = status.HTTP_200_OK if response.ready else status.HTTP_503_SERVICE_UNAVAILABLE
    if response.ready:
        logger.debug("Readiness check passed: %s", response.checks)
    else:
        logger.warning("Readiness check failed: %s", response.checks)
    return JSONResponse(content=response.model_dump(mode="json"), status_code=status_code)
