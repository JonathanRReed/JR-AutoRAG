"""Security middleware for FastAPI.

Integrates authentication, rate limiting, audit logging, and request guards
into a cohesive security layer for production deployments.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Callable, Optional
from functools import wraps

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .auth import get_auth, APIKeyAuth
from .rate_limiter import get_rate_limiter, RateLimiter
from .audit import get_audit_log, AuditAction


# =============================================================================
# Configuration
# =============================================================================

# Safe default origins (localhost only)
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
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

# Scope requirements per route prefix
ROUTE_SCOPES = {
    "/query": "read",
    "/documents": "write",
    "/config": "admin",
    "/evaluation": "read",
    "/admin": "admin",
    "/api/keys": "admin",
}

# Request size limits (bytes)
MAX_REQUEST_SIZE = int(os.environ.get("AUTORAG_MAX_REQUEST_SIZE", 50 * 1024 * 1024))  # 50MB default

# Per-route timeout configurations (seconds)
ROUTE_TIMEOUTS = {
    "/query": 120,
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


# =============================================================================
# Authentication Dependency
# =============================================================================

async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[str]:
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
    
    # Determine required scope based on route/method
    required_scope = None
    if path.startswith("/documents"):
        if request.method.upper() in {"GET", "HEAD", "OPTIONS"}:
            required_scope = "read"
        else:
            required_scope = "write"
    else:
        for prefix, scope in ROUTE_SCOPES.items():
            if path.startswith(prefix):
                required_scope = scope
                break
    
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
        # Skip rate limiting for health checks
        if request.url.path in {"/healthz", "/readyz"}:
            return await call_next(request)
        
        # Determine rate limit key (API key > IP)
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            rate_key = f"key:{api_key[:16]}"
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
    """Middleware that adds timeout information to request state."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        
        # Determine timeout for this route
        timeout = ROUTE_TIMEOUTS.get("default", 60)
        for prefix, route_timeout in ROUTE_TIMEOUTS.items():
            if prefix != "default" and path.startswith(prefix):
                timeout = route_timeout
                break
        
        # Store timeout in request state for handlers to use
        request.state.timeout = timeout
        request.state.start_time = time.time()
        
        response = await call_next(request)
        
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
    
    # Log configuration
    auth = get_auth()
    rate_limiter = get_rate_limiter()
    
    print(f"Security configured:")
    print(f"  - Authentication: {'enabled' if auth.require_auth() else 'disabled'}")
    print(f"  - Rate limiting: {'enabled' if rate_limiter.enabled else 'disabled'}")
    print(f"  - Max request size: {MAX_REQUEST_SIZE // (1024*1024)}MB")
    print(f"  - Allowed origins: {get_allowed_origins()}")
    print(f"  - Exposed mode: {is_exposed_mode()}")


__all__ = [
    "verify_api_key",
    "RateLimitMiddleware",
    "RequestSizeLimitMiddleware",
    "TimeoutMiddleware",
    "SecurityHeadersMiddleware",
    "configure_security",
    "get_allowed_origins",
    "is_exposed_mode",
    "PUBLIC_PATHS",
    "ROUTE_SCOPES",
    "MAX_REQUEST_SIZE",
    "ROUTE_TIMEOUTS",
]
