"""FastAPI routers for JR AutoRAG."""

from . import health, config, documents, query, evaluation, monitoring, providers

__all__ = ["health", "config", "documents", "query", "evaluation", "monitoring", "providers"]
