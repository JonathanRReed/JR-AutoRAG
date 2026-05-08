"""Security posture schemas for install readiness."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SecurityCheckStatus = Literal["pass", "warn", "fail"]
SecurityPostureLevel = Literal["local_only", "client_ready", "needs_attention", "unsafe"]


class SecurityPostureCheck(BaseModel):
    id: str
    status: SecurityCheckStatus
    message: str
    detail: str | None = None
    remediation: str | None = None


class SecurityPostureSettings(BaseModel):
    auth_enabled: bool
    api_keys_configured: bool
    exposed_mode: bool
    rate_limit_enabled: bool
    allowed_origin_count: int = Field(ge=0)
    wildcard_cors: bool
    docs_public: bool


class SecurityPostureResponse(BaseModel):
    level: SecurityPostureLevel
    summary: str
    settings: SecurityPostureSettings
    checks: list[SecurityPostureCheck]
    recommendations: list[str]
