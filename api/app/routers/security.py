"""Security posture endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..core.security_posture import build_security_posture
from ..schemas.security import SecurityPostureResponse

router = APIRouter(prefix="/security", tags=["security"])


@router.get("/posture", response_model=SecurityPostureResponse)
async def get_security_posture() -> SecurityPostureResponse:
    """Return redacted install security posture for operators."""
    return build_security_posture()
