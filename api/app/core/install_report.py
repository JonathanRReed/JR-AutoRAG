"""Redacted install report builder for client handoff."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..schemas.health import ReadinessResponse
from ..schemas.install import (
    InstallReportAction,
    InstallReportCorpus,
    InstallReportEvidence,
    InstallReportResponse,
)
from ..schemas.security import SecurityPostureResponse
from ..services import ServiceContainer


def _safe_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _collect_corpus(container: ServiceContainer) -> InstallReportCorpus:
    docs = container.document_store.list()
    parser_counts: dict[str, int] = {}
    low_confidence = 0
    processing_errors = 0
    for doc in docs:
        parser = str(doc.metadata.get("parser_provider", "native"))
        parser_counts[parser] = parser_counts.get(parser, 0) + 1
        try:
            confidence = float(doc.metadata.get("parser_confidence", "1") or 1)
        except ValueError:
            confidence = 1.0
        if confidence < 0.7:
            low_confidence += 1
        if doc.metadata.get("processing_status") == "error":
            processing_errors += 1

    audit_context = container.orchestrator.get_eval_audit_context()
    manifest = _safe_mapping(audit_context.get("corpus"))
    return InstallReportCorpus(
        document_count=len(docs),
        chunk_count=int(manifest.get("chunk_count") or 0),
        fingerprint=manifest.get("fingerprint"),
        parser_counts=parser_counts,
        low_confidence_documents=low_confidence,
        processing_errors=processing_errors,
    )


def _build_evidence(
    corpus: InstallReportCorpus,
    evaluations: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> list[InstallReportEvidence]:
    latest_eval = evaluations[0] if evaluations else {}
    eval_sha = latest_eval.get("report_sha256")
    return [
        InstallReportEvidence(
            id="doctor",
            title="Install doctor",
            status="present",
            detail="Run bun run doctor -- --json on the target machine before handoff.",
            endpoint=None,
        ),
        InstallReportEvidence(
            id="security_posture",
            title="Security posture",
            status="present",
            detail="Redacted auth, exposure, CORS, docs, header, and rate-limit checks.",
            endpoint="/security/posture",
        ),
        InstallReportEvidence(
            id="readiness",
            title="Runtime readiness",
            status="present",
            detail="Typed API readiness probe for service, document store, retrieval, models, and provider state.",
            endpoint="/readyz",
        ),
        InstallReportEvidence(
            id="corpus_manifest",
            title="Corpus manifest",
            status="present" if corpus.document_count else "missing",
            detail=f"{corpus.document_count} document(s), {corpus.chunk_count} chunk(s).",
        ),
        InstallReportEvidence(
            id="quality_receipt",
            title="Golden evaluation receipt",
            status="present" if eval_sha else "missing",
            detail="Latest golden run report with digest." if eval_sha else "Run a golden evaluation before client handoff.",
            endpoint=f"/evaluation/runs/{latest_eval.get('run_id')}/report" if latest_eval.get("run_id") else None,
            artifact_path=latest_eval.get("report_path"),
            sha256=eval_sha,
        ),
        InstallReportEvidence(
            id="retrieval_artifacts",
            title="Retrieval artifacts",
            status="present"
            if artifacts.get("graph_rag_status") == "ready" or artifacts.get("raptor_status") == "ready"
            else "warn",
            detail=(
                f"GraphRAG={artifacts.get('graph_rag_status', 'unknown')}, "
                f"RAPTOR={artifacts.get('raptor_status', 'unknown')}"
            ),
            endpoint="/api/artifacts/status",
        ),
    ]


def _build_actions(
    readiness: ReadinessResponse,
    security: SecurityPostureResponse,
    corpus: InstallReportCorpus,
    evaluations: list[dict[str, Any]],
) -> list[InstallReportAction]:
    actions: list[InstallReportAction] = []
    if any(check.status == "fail" for check in security.checks):
        actions.append(InstallReportAction(
            id="fix-security",
            title="Fix failed security checks",
            priority="high",
            detail="Resolve failed auth, CORS, exposure, docs, headers, or rate-limit checks before client-network install.",
            endpoint="/security/posture",
        ))
    if readiness.level == "not_ready":
        actions.append(InstallReportAction(
            id="fix-readiness",
            title="Restore runtime readiness",
            priority="high",
            detail="Resolve failed runtime checks before relying on the install.",
            endpoint="/readyz",
        ))
    if corpus.document_count == 0:
        actions.append(InstallReportAction(
            id="ingest-corpus",
            title="Ingest a representative corpus",
            priority="high",
            detail="Add representative client documents, then rerun retrieval and evaluation checks.",
        ))
    if corpus.processing_errors:
        actions.append(InstallReportAction(
            id="fix-ingestion-errors",
            title="Fix ingestion errors",
            priority="high",
            detail=f"{corpus.processing_errors} document(s) failed processing.",
        ))
    if not evaluations or not evaluations[0].get("report_sha256"):
        actions.append(InstallReportAction(
            id="run-golden-eval",
            title="Run a golden evaluation",
            priority="medium",
            detail="Create a digest-backed quality receipt before handoff.",
            endpoint="/evaluation/batch/{set_name}",
        ))
    if security.level == "needs_attention":
        actions.append(InstallReportAction(
            id="harden-client-exposure",
            title="Harden client exposure settings",
            priority="medium",
            detail="Enable API-key auth and rate limiting before exposing beyond localhost.",
            command="AUTORAG_AUTH_ENABLED=true AUTORAG_RATE_LIMIT_ENABLED=true",
        ))
    return actions


def build_install_report(
    *,
    container: ServiceContainer,
    readiness: ReadinessResponse,
    security: SecurityPostureResponse,
    evaluations: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> InstallReportResponse:
    """Build a single redacted report for client install evidence."""
    corpus = _collect_corpus(container)
    policy = container.local_first.describe()
    evidence = _build_evidence(corpus, evaluations, artifacts)
    actions = _build_actions(readiness, security, corpus, evaluations)

    has_security_failure = any(check.status == "fail" for check in security.checks)
    if has_security_failure or readiness.level == "not_ready":
        status = "blocked"
        summary = "Install is blocked by failed readiness or security checks."
    elif actions:
        status = "warn"
        summary = "Install is usable for local evaluation, but client handoff still needs evidence work."
    else:
        status = "ready"
        summary = "Install has current readiness, security posture, corpus evidence, and quality receipts."

    return InstallReportResponse(
        generated_at=datetime.now(UTC).isoformat(),
        status=status,
        summary=summary,
        readiness=readiness,
        security=security,
        policy=policy,
        corpus=corpus,
        evaluations=evaluations,
        artifacts=artifacts,
        evidence=evidence,
        actions=actions,
        redaction={
            "secrets": "API keys, tokens, passwords, credentials, and secret-shaped config keys are omitted or redacted.",
            "scope": "Report includes operational metadata only; ingested document text and retrieved chunks are not embedded.",
        },
    )
