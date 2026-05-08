"""Install readiness report schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .health import ReadinessResponse
from .security import SecurityPostureResponse


InstallReportStatus = Literal["ready", "warn", "blocked"]


class InstallReportAction(BaseModel):
    """Operator action needed before a client handoff."""

    id: str
    title: str
    priority: Literal["high", "medium", "low"]
    detail: str
    command: str | None = None
    endpoint: str | None = None


class InstallReportEvidence(BaseModel):
    """Auditable evidence references for installer handoff."""

    id: str
    title: str
    status: Literal["present", "missing", "warn"]
    detail: str
    endpoint: str | None = None
    artifact_path: str | None = None
    sha256: str | None = None


class InstallReportCorpus(BaseModel):
    """Corpus state included in the handoff report."""

    document_count: int = 0
    chunk_count: int = 0
    fingerprint: str | None = None
    parser_counts: dict[str, int] = Field(default_factory=dict)
    low_confidence_documents: int = 0
    processing_errors: int = 0


class InstallReportResponse(BaseModel):
    """Redacted install report for B2B local deployments."""

    schema_version: Literal["install_report_v1"] = "install_report_v1"
    generated_at: str
    product: str = "JR AutoRAG"
    status: InstallReportStatus
    summary: str
    readiness: ReadinessResponse
    security: SecurityPostureResponse
    policy: dict[str, Any]
    corpus: InstallReportCorpus
    evaluations: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    evidence: list[InstallReportEvidence] = Field(default_factory=list)
    actions: list[InstallReportAction] = Field(default_factory=list)
    redaction: dict[str, str] = Field(default_factory=dict)
