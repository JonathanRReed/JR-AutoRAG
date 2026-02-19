"""JR AutoRAG API - Production-ready FastAPI application.

Security features:
- Safe CORS defaults (localhost only, configurable via AUTORAG_ALLOWED_ORIGINS)
- API key authentication (optional, enable via AUTORAG_AUTH_ENABLED=true)
- Rate limiting (optional, enable via AUTORAG_RATE_LIMIT_ENABLED=true)
- Request size limits, timeouts, and security headers

Environment variables:
- AUTORAG_ALLOWED_ORIGINS: Comma-separated list of allowed origins
- AUTORAG_AUTH_ENABLED: Enable API key authentication (true/false)
- AUTORAG_API_KEYS: Comma-separated API keys for authentication
- AUTORAG_EXPOSE: Allow binding to non-localhost (true/false)
- AUTORAG_MAX_REQUEST_SIZE: Maximum request body size in bytes
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.audit import AuditAction, AuditEntry, get_audit_log
from .core.providers import close_shared_client
from .core.security_middleware import (
    configure_security,
    get_allowed_origins,
    is_exposed_mode,
    verify_api_key,
)
from .routers import (
    artifact_routes,
    cache_routes,
    config,
    documents,
    evaluation,
    health,
    metrics_routes,
    monitoring,
    providers,
    query,
    ragfuzz_audit,
    traces,
)
from .services import get_container

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Initialize services on startup
    logger.info("=" * 60)
    logger.info("JR AutoRAG API - Starting...")
    logger.info("=" * 60)

    # Log startup in audit log
    audit_log = get_audit_log()
    audit_log.log(AuditEntry(
        timestamp=datetime.utcnow(),
        action=AuditAction.SYSTEM,
        details={"event": "startup", "version": app.version},
    ))

    # Initialize service container
    logger.info("Initializing Application Services...")
    container = get_container()
    logger.info("Application Services Initialized.")

    # Log security configuration
    logger.info("Security Configuration:")
    logger.info(f"  Allowed Origins: {get_allowed_origins()}")
    logger.info(f"  Exposed Mode: {is_exposed_mode()}")
    logger.info(f"  Auth Enabled: {os.environ.get('AUTORAG_AUTH_ENABLED', 'false')}")
    logger.info("")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Orchestrator...")
    if container.orchestrator:
        await container.orchestrator.stop()

    # Close shared HTTP client
    logger.info("Closing HTTP connection pool...")
    await close_shared_client()

    # Log shutdown in audit log
    audit_log.log(AuditEntry(
        timestamp=datetime.utcnow(),
        action=AuditAction.SYSTEM,
        details={"event": "shutdown"},
    ))

    logger.info("JR AutoRAG API - Shutdown complete.")


# Create FastAPI app
app = FastAPI(
    title="JR AutoRAG API",
    version="3.0.0",
    description="Production-ready RAG workbench with security, evaluation gates, and reproducibility",
    lifespan=lifespan,
)

# =============================================================================
# CORS Configuration - Safe Defaults
# =============================================================================

# Use safe defaults instead of wildcard origins
allowed_origins = get_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Limit", "X-Response-Time"],
)

# =============================================================================
# Security Middleware
# =============================================================================

# Configure security middleware (rate limiting, request guards, security headers)
configure_security(app)

# =============================================================================
# Routers
# =============================================================================

# Health endpoints (public, no auth required)
app.include_router(health.router)

# Protected endpoints (auth optional based on configuration)
app.include_router(
    config.router,
    prefix="/config",
    tags=["config"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    documents.router,
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    query.router,
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    evaluation.router,
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    monitoring.router,
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    providers.router,
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    traces.router,
    prefix="/api",
    tags=["traces"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    artifact_routes.router,
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    cache_routes.router,
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    metrics_routes.router,
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    ragfuzz_audit.router,
    dependencies=[Depends(verify_api_key)],
)


# =============================================================================
# Root Endpoint
# =============================================================================

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "JR AutoRAG API",
        "status": "ok",
        "version": app.version,
        "security": {
            "auth_enabled": os.environ.get("AUTORAG_AUTH_ENABLED", "false").lower() == "true",
            "cors_origins": len(allowed_origins),
        },
    }
