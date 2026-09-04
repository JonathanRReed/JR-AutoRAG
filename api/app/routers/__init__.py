"""FastAPI routers for JR AutoRAG."""

from . import (
    cache_routes,
    config,
    documents,
    evaluation,
    health,
    install,
    monitoring,
    providers,
    query,
    security,
    traces,
)

__all__ = [
    "cache_routes",
    "health",
    "config",
    "documents",
    "query",
    "evaluation",
    "install",
    "monitoring",
    "providers",
    "security",
    "traces",
]
