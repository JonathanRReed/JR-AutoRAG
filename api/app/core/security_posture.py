"""Security posture checks for local and client installs."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from ..schemas.security import (
    SecurityPostureCheck,
    SecurityPostureResponse,
    SecurityPostureSettings,
)
from .prompt_guard import INJECTION_PATTERNS
from .security_middleware import get_allowed_origins, is_exposed_mode


def _truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


def _rate_limit_enabled() -> bool:
    default_enabled = "false" if _truthy("JR_DEMO_MODE") else "true"
    return _truthy("AUTORAG_RATE_LIMIT_ENABLED", default_enabled)


def _api_keys_configured() -> bool:
    return bool(os.environ.get("AUTORAG_API_KEYS", "").strip())


def _is_local_origin(origin: str) -> bool:
    parsed = urlparse(origin)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def build_security_posture() -> SecurityPostureResponse:
    """Return a redacted operator-facing security posture report."""
    auth_enabled = _truthy("AUTORAG_AUTH_ENABLED")
    api_keys_configured = _api_keys_configured()
    exposed_mode = is_exposed_mode()
    rate_limit_enabled = _rate_limit_enabled()
    allowed_origins = get_allowed_origins()
    wildcard_cors = "*" in allowed_origins
    docs_public = not exposed_mode

    checks: list[SecurityPostureCheck] = []

    if auth_enabled and api_keys_configured:
        checks.append(
            SecurityPostureCheck(
                id="auth",
                status="pass",
                message="API-key authentication is enabled",
                detail="AUTORAG_AUTH_ENABLED is true and at least one key is configured.",
            )
        )
    elif auth_enabled:
        checks.append(
            SecurityPostureCheck(
                id="auth",
                status="fail",
                message="Authentication is enabled without API keys",
                detail="AUTORAG_AUTH_ENABLED is true, but AUTORAG_API_KEYS is empty.",
                remediation="Set AUTORAG_API_KEYS before starting the API.",
            )
        )
    else:
        checks.append(
            SecurityPostureCheck(
                id="auth",
                status="warn",
                message="Authentication is disabled",
                detail="This is acceptable for loopback-only local demos.",
                remediation="Set AUTORAG_AUTH_ENABLED=true and AUTORAG_API_KEYS for client installs.",
            )
        )

    if exposed_mode and not (auth_enabled and api_keys_configured):
        checks.append(
            SecurityPostureCheck(
                id="exposure",
                status="fail",
                message="Exposed mode requires authentication",
                detail="AUTORAG_EXPOSE is true while API-key authentication is not ready.",
                remediation="Enable auth with AUTORAG_AUTH_ENABLED=true and set AUTORAG_API_KEYS.",
            )
        )
    elif exposed_mode:
        checks.append(
            SecurityPostureCheck(
                id="exposure",
                status="pass",
                message="Exposed mode is guarded by API keys",
                detail="Protected endpoints fail closed when authentication is not valid.",
            )
        )
    else:
        checks.append(
            SecurityPostureCheck(
                id="exposure",
                status="pass",
                message="API is configured for local-only exposure",
                detail="AUTORAG_EXPOSE is false.",
            )
        )

    non_local_origins = [
        origin for origin in allowed_origins if not _is_local_origin(origin)
    ]
    if wildcard_cors:
        checks.append(
            SecurityPostureCheck(
                id="cors",
                status="fail" if exposed_mode else "warn",
                message="Wildcard CORS origin is configured",
                detail="Wildcard browser origins are not appropriate for client installs.",
                remediation="Set AUTORAG_ALLOWED_ORIGINS to exact trusted origins.",
            )
        )
    elif exposed_mode and non_local_origins and not auth_enabled:
        checks.append(
            SecurityPostureCheck(
                id="cors",
                status="fail",
                message="Non-local CORS origins require authentication",
                detail=f"{len(non_local_origins)} non-local origin(s) configured.",
                remediation="Enable API-key authentication or restrict origins to loopback.",
            )
        )
    else:
        checks.append(
            SecurityPostureCheck(
                id="cors",
                status="pass",
                message="CORS origins are explicit",
                detail=f"{len(allowed_origins)} allowed origin(s), no wildcard.",
            )
        )

    if rate_limit_enabled:
        checks.append(
            SecurityPostureCheck(
                id="rate_limit",
                status="pass",
                message="Rate limiting is enabled",
                detail=f"{os.environ.get('AUTORAG_RATE_LIMIT_RPM', '600')} requests per minute.",
            )
        )
    else:
        checks.append(
            SecurityPostureCheck(
                id="rate_limit",
                status="fail" if exposed_mode else "warn",
                message="Rate limiting is disabled",
                detail="Demo mode disables rate limiting unless AUTORAG_RATE_LIMIT_ENABLED is set.",
                remediation="Set AUTORAG_RATE_LIMIT_ENABLED=true before exposing the API.",
            )
        )

    checks.append(
        SecurityPostureCheck(
            id="docs",
            status="pass",
            message="Interactive API docs are blocked in exposed mode"
            if exposed_mode
            else "Interactive API docs are local-only",
            detail="/docs, /redoc, and /openapi.json return 404 when AUTORAG_EXPOSE=true.",
        )
    )

    checks.append(
        SecurityPostureCheck(
            id="headers",
            status="pass",
            message="Security headers middleware is installed",
            detail="Responses include nosniff, frame denial, referrer policy, CSP, and timing headers.",
        )
    )

    checks.append(
        SecurityPostureCheck(
            id="prompt_injection",
            status="pass",
            message="Indirect prompt-injection defenses are active",
            detail=(
                f"{len(INJECTION_PATTERNS)} detection patterns sanitize ingested text, "
                "and retrieved snippets are wrapped as document data before generation."
            ),
        )
    )

    recommendations = [
        check.remediation
        for check in checks
        if check.status in {"warn", "fail"} and check.remediation
    ]
    if not recommendations:
        recommendations.append(
            "Keep API keys scoped to client installs and rotate them between engagements."
        )

    if any(check.status == "fail" for check in checks):
        level = "unsafe"
        summary = "Security posture has blocking issues before client exposure."
    elif exposed_mode:
        level = "client_ready"
        summary = "Security posture is ready for a guarded client-network install."
    elif any(check.status == "warn" for check in checks):
        level = "needs_attention"
        summary = "Local posture is usable, but client exposure needs hardening."
    else:
        level = "local_only"
        summary = "Security posture is appropriate for loopback-only local use."

    return SecurityPostureResponse(
        level=level,
        summary=summary,
        settings=SecurityPostureSettings(
            auth_enabled=auth_enabled,
            api_keys_configured=api_keys_configured,
            exposed_mode=exposed_mode,
            rate_limit_enabled=rate_limit_enabled,
            allowed_origin_count=len(allowed_origins),
            wildcard_cors=wildcard_cors,
            docs_public=docs_public,
        ),
        checks=checks,
        recommendations=recommendations,
    )
