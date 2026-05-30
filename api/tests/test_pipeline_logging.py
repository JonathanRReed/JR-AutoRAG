from __future__ import annotations

from datetime import UTC, datetime

from app.core.orchestrator import _pipeline_step_log_record
from app.core.telemetry import PipelineStep


def test_pipeline_step_log_record_omits_sensitive_details():
    secret = "SSN-123-45-6789 confidential acquisition target: ProjectNightfall"
    now = datetime.now(UTC)
    step = PipelineStep(
        name="planning",
        started_at=now,
        completed_at=now,
        duration_ms=12.5,
        status="ok",
        details={
            "queries": [secret],
            "iterations": [{"query": secret, "refined_query": secret}],
            "conflicts_summary": [f"retrieved text {secret}"],
            "outline": f"context-derived outline {secret}",
        },
    )

    log_record = _pipeline_step_log_record("trace-123", step)

    assert log_record == {
        "event": "pipeline_step",
        "trace_id": "trace-123",
        "name": "planning",
        "duration_ms": 12.5,
        "status": "ok",
    }
    assert "details" not in log_record
    assert secret not in repr(log_record)
