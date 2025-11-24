"""FastAPI routers for JR AutoRAG."""

from . import config, documents, evaluation, health, monitoring, providers, query

__all__ = ["health", "config", "documents", "query", "evaluation", "monitoring", "providers"]
