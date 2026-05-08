"""Install report endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..core.install_report import build_install_report
from ..core.security_posture import build_security_posture
from ..routers.evaluation import get_eval_run_store
from ..schemas.health import ReadinessResponse
from ..schemas.install import InstallReportResponse
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/install", tags=["install"])


@router.get("/report", response_model=InstallReportResponse)
async def get_install_report(container: ServiceContainer = Depends(get_container)) -> InstallReportResponse:
    """Return a redacted client handoff report for the current local install."""
    readiness = ReadinessResponse.model_validate(container.orchestrator.get_readiness_status())
    security = build_security_posture()
    evaluations = get_eval_run_store().list_runs(limit=10)
    artifacts = container.orchestrator.get_artifact_build_status()
    return build_install_report(
        container=container,
        readiness=readiness,
        security=security,
        evaluations=evaluations,
        artifacts=artifacts,
    )
