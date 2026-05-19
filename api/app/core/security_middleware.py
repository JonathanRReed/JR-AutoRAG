"""Security middleware for FastAPI.

Integrates authentication, rate limiting, audit logging, and request guards
into a cohesive security layer for production deployments.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from .audit import get_audit_log
from .auth import get_auth
from .rate_limiter import get_rate_limiter

# =============================================================================
# Configuration
# =============================================================================

# Safe default origins (localhost only)
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:5173",
]

# Paths that don't require authentication
PUBLIC_PATHS = {
    "/",
    "/healthz",
    "/readyz",
    "/docs",
    "/openapi.json",
    "/redoc",
}

DOCS_PATHS = {"/docs", "/openapi.json", "/redoc"}

# Scope requirements per route prefix
ROUTE_SCOPES = {
    "/api/artifacts/build": "admin",
    "/api/cache/clear": "admin",
    "/api/cache/rebuild": "admin",
    "/rag/audit": "admin",
    "/query": "read",
    "/documents": "write",
    "/config": "admin",
    "/evaluation": "read",
    "/providers": "read",
    "/security": "read",
    "/install": "read",
    "/monitoring": "read",
    "/api/traces": "read",
    "/api/artifacts": "read",
    "/api/cache": "read",
    "/admin": "admin",
    "/api/keys": "admin",
}

# Request size limits (bytes)
MAX_REQUEST_SIZE = int(os.environ.get("AUTORAG_MAX_REQUEST_SIZE", 50 * 1024 * 1024))  # 50MB default

# Per-route timeout configurations (seconds)
ROUTE_TIMEOUTS = {
    "/query": 300,
    "/query/stream": 300,
    "/documents/upload": 600,
    "/evaluation/run": 900,
    "default": 60,
}


# =============================================================================
# API Key Header
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_allowed_origins() -> list[str]:
    """Get allowed CORS origins from environment or defaults."""
    env_origins = os.environ.get("AUTORAG_ALLOWED_ORIGINS", "")
    if env_origins:
        return [o.strip() for o in env_origins.split(",") if o.strip()]
    return DEFAULT_ALLOWED_ORIGINS


def is_exposed_mode() -> bool:
    """Check if server is in exposed mode (non-localhost binding)."""
    return os.environ.get("AUTORAG_EXPOSE", "false").lower() in ("true", "1", "yes")


def _resolve_required_scope(path: str, method: str) -> str | None:
    """Resolve required scope based on request path and method."""
    if path.startswith("/documents"):
        return "read" if method.upper() in {"GET", "HEAD", "OPTIONS"} else "write"
    for prefix, scope in ROUTE_SCOPES.items():
        if path.startswith(prefix):
            return scope
    return None


def _resolve_route_timeout(path: str) -> int:
    """Resolve route timeout using the most specific prefix first."""
    timeout = ROUTE_TIMEOUTS.get("default", 60)
    for prefix, route_timeout in sorted(
        ((key, value) for key, value in ROUTE_TIMEOUTS.items() if key != "default"),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if path.startswith(prefix):
            return route_timeout
    return timeout


# =============================================================================
# Authentication Dependency
# =============================================================================

async def verify_api_key(
    request: Request,
    api_key: str | None = Depends(api_key_header),
) -> str | None:
    """FastAPI dependency for API key verification.

    Returns:
        User identifier if authenticated, None if auth not required

    Raises:
        HTTPException: If auth required but key invalid
    """
    auth = get_auth()
    path = request.url.path

    # Default user context to None for downstream consumers
    request.state.user_id = None
    request.state.scopes = []

    # Skip auth for public paths
    if path in PUBLIC_PATHS:
        return None

    if is_exposed_mode() and not auth.require_auth():
        raise HTTPException(
            status_code=503,
            detail=(
                "Refusing unauthenticated access while AUTORAG_EXPOSE=true. "
                "Enable AUTORAG_AUTH_ENABLED and configure AUTORAG_API_KEYS."
            ),
        )

    # If auth is not enabled, allow
    if not auth.require_auth():
        return None

    # Fail closed if auth enabled but no keys configured
    if not auth.has_keys():
        raise HTTPException(
            status_code=500,
            detail="Authentication is enabled but no API keys are configured. Set AUTORAG_API_KEYS.",
        )

    # Auth is required - verify key
    if not api_key:
        audit_log = get_audit_log()
        audit_log.log_auth(
            success=False,
            ip_address=request.client.host if request.client else None,
            reason="Missing API key",
        )
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    required_scope = _resolve_required_scope(path, request.method)

    if not auth.verify(api_key, required_scope=required_scope):
        audit_log = get_audit_log()
        audit_log.log_auth(
            success=False,
            ip_address=request.client.host if request.client else None,
            reason=f"Invalid key or missing scope: {required_scope}",
        )
        raise HTTPException(
            status_code=403,
            detail="Invalid API key or insufficient permissions.",
        )

    scopes = auth.get_scopes(api_key)
    # Log successful auth
    audit_log = get_audit_log()
    user_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    audit_log.log_auth(
        success=True,
        user_id=user_id + "...",  # Log short hash for identification
        ip_address=request.client.host if request.client else None,
    )

    request.state.user_id = user_id
    request.state.scopes = scopes
    return user_id  # Return short hash as user identifier


# =============================================================================
# Rate Limiting Middleware
# =============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces rate limiting per client."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.limiter = get_rate_limiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and CORS preflight requests
        if request.url.path in {"/healthz", "/readyz"} or request.method == "OPTIONS":
            return await call_next(request)

        # Determine rate limit key (API key > IP)
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            # Use hash of API key instead of plaintext prefix for rate limiting
            rate_key = f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        else:
            client_ip = request.client.host if request.client else "unknown"
            rate_key = f"ip:{client_ip}"

        # Check rate limit
        if not self.limiter.allow(rate_key):
            wait_time = self.limiter.get_wait_time(rate_key)
            return Response(
                content=f"Rate limit exceeded. Retry after {wait_time:.1f} seconds.",
                status_code=429,
                headers={
                    "Retry-After": str(int(wait_time) + 1),
                    "X-RateLimit-Remaining": "0",
                },
            )

        # Add rate limit headers to response
        response = await call_next(request)
        remaining = self.limiter.get_remaining(rate_key)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Limit"] = str(self.limiter._requests_per_minute)

        return response


# =============================================================================
# Exposed Mode Docs Guard
# =============================================================================

class ExposedDocsBlockerMiddleware(BaseHTTPMiddleware):
    """Middleware that disables interactive docs when the API is exposed."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in DOCS_PATHS and is_exposed_mode():
            return JSONResponse(
                {"detail": "Interactive API docs are disabled while AUTORAG_EXPOSE=true."},
                status_code=404,
            )
        return await call_next(request)


# =============================================================================
# Request Size Limit Middleware
# =============================================================================

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces request body size limits."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                size = int(content_length)
                if size > MAX_REQUEST_SIZE:
                    return Response(
                        content=f"Request too large. Maximum size is {MAX_REQUEST_SIZE // (1024*1024)}MB.",
                        status_code=413,
                    )
            except ValueError:
                pass

        return await call_next(request)


# =============================================================================
# Timeout Middleware
# =============================================================================

class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces route-level timeouts."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Determine timeout for this route
        timeout = _resolve_route_timeout(path)

        # Store timeout in request state for handlers to use
        request.state.timeout = timeout
        request.state.start_time = time.time()

        try:
            response = await asyncio.wait_for(call_next(request), timeout=timeout)
        except TimeoutError:
            return Response(
                content=f"Request timed out after {timeout} seconds.",
                status_code=504,
            )

        # Add timing header
        elapsed = time.time() - request.state.start_time
        response.headers["X-Response-Time"] = f"{elapsed:.3f}s"

        return response


# =============================================================================
# Security Headers Middleware
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that adds security headers to responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # CSP for API responses
        if not request.url.path.startswith("/docs"):
            response.headers["Content-Security-Policy"] = "default-src 'none'"

        return response


# =============================================================================
# Apply Security to App
# =============================================================================

def configure_security(app: FastAPI, *, strict: bool = False) -> None:
    """Configure all security middleware for the application.

    Args:
        app: FastAPI application instance
        strict: If True, enable stricter security (for production)
    """
    # Add middleware in reverse order (last added runs first)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(TimeoutMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(ExposedDocsBlockerMiddleware)

    # Log configuration
    auth = get_auth()
    rate_limiter = get_rate_limiter()

    print("Security configured:")
    print(f"  - Authentication: {'enabled' if auth.require_auth() else 'disabled'}")
    print(f"  - Rate limiting: {'enabled' if rate_limiter.enabled else 'disabled'}")
    print(f"  - Max request size: {MAX_REQUEST_SIZE // (1024*1024)}MB")
    print(f"  - Allowed origins: {get_allowed_origins()}")
    print(f"  - Exposed mode: {is_exposed_mode()}")


__all__ = [
    "verify_api_key",
    "RateLimitMiddleware",
    "ExposedDocsBlockerMiddleware",
    "RequestSizeLimitMiddleware",
    "TimeoutMiddleware",
    "SecurityHeadersMiddleware",
    "configure_security",
    "get_allowed_origins",
    "is_exposed_mode",
    "_resolve_route_timeout",
    "PUBLIC_PATHS",
    "ROUTE_SCOPES",
    "MAX_REQUEST_SIZE",
    "ROUTE_TIMEOUTS",
    "DOCS_PATHS",
]
