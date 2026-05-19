"""Typed health and readiness response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


CheckStatus = Literal["ok", "warn", "fail"]


class ReadinessCheck(BaseModel):
    """One runtime readiness check."""

    status: CheckStatus
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ReadinessResponse(BaseModel):
    """Runtime readiness report for operators and deployment probes."""

    ready: bool
    level: Literal["ready", "degraded", "not_ready"]
    checks: dict[str, ReadinessCheck]
