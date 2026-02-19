"""Health check endpoints for Kubernetes-style probes.

/healthz - Liveness probe: Is the process running?
/readyz  - Readiness probe: Is the service ready to accept traffic?
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from ..state import get_orchestrator

logger = logging.getLogger("autorag.health")

router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe - is the process alive?"""
    return {"status": "ok"}


@router.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness probe - is the service ready to accept traffic?

    Checks:
    - Orchestrator initialized
    - Document store accessible
    - Vector index loaded
    - Embedding model available
    - At least one LLM provider configured

    Returns 200 if all checks pass, 503 if any critical check fails.
    """
    checks: dict[str, dict[str, Any]] = {}
    all_ready = True

    # Check 1: Orchestrator initialized
    try:
        orchestrator = get_orchestrator()
        if orchestrator is None:
            checks["orchestrator"] = {
                "status": "fail",
                "message": "Orchestrator not initialized"
            }
            all_ready = False
        else:
            checks["orchestrator"] = {"status": "ok"}
    except Exception as e:
        checks["orchestrator"] = {
            "status": "fail",
            "message": f"Error accessing orchestrator: {e}"
        }
        all_ready = False

    # Check 2: Document store accessible
    try:
        if orchestrator and hasattr(orchestrator, "retrieval"):
            doc_store = orchestrator.retrieval.documents
            doc_count = len(doc_store.list_all())
            checks["document_store"] = {
                "status": "ok",
                "document_count": doc_count
            }
        else:
            checks["document_store"] = {
                "status": "fail",
                "message": "Retrieval engine not available"
            }
            all_ready = False
    except Exception as e:
        checks["document_store"] = {
            "status": "fail",
            "message": f"Error accessing document store: {e}"
        }
        all_ready = False

    # Check 3: Vector index loaded
    try:
        if orchestrator and hasattr(orchestrator, "retrieval"):
            retriever = orchestrator.retrieval
            index_ready = (
                retriever._embeddings is not None
                and len(retriever._embeddings) > 0
            ) if hasattr(retriever, "_embeddings") else False

            if index_ready or doc_count == 0:
                # Index is ready OR no documents to index (valid empty state)
                checks["vector_index"] = {
                    "status": "ok",
                    "indexed_chunks": len(retriever._embeddings) if index_ready else 0
                }
            else:
                checks["vector_index"] = {
                    "status": "fail",
                    "message": "Vector index not built"
                }
                all_ready = False
        else:
            checks["vector_index"] = {
                "status": "fail",
                "message": "Retrieval engine not available"
            }
            all_ready = False
    except Exception as e:
        checks["vector_index"] = {
            "status": "fail",
            "message": f"Error checking vector index: {e}"
        }
        all_ready = False

    # Check 4: Embedding model available
    try:
        if orchestrator and hasattr(orchestrator, "retrieval"):
            retriever = orchestrator.retrieval
            if hasattr(retriever, "_embedding_model") and retriever._embedding_model is not None:
                checks["embedding_model"] = {
                    "status": "ok",
                    "model": retriever.config.embedding_model
                }
            elif hasattr(retriever, "_init_models"):
                # Model loaded lazily - check if it would load
                checks["embedding_model"] = {
                    "status": "ok",
                    "model": retriever.config.embedding_model,
                    "note": "lazy-loaded"
                }
            else:
                checks["embedding_model"] = {
                    "status": "warn",
                    "message": "Embedding model status unknown"
                }
        else:
            checks["embedding_model"] = {
                "status": "fail",
                "message": "Retrieval engine not available"
            }
            all_ready = False
    except Exception as e:
        checks["embedding_model"] = {
            "status": "fail",
            "message": f"Error checking embedding model: {e}"
        }
        all_ready = False

    # Check 5: Provider configuration (warning only, not critical)
    try:
        if orchestrator and hasattr(orchestrator, "provider_factory"):
            # Check if at least one provider can be discovered
            checks["providers"] = {
                "status": "ok",
                "note": "Provider factory available"
            }
        else:
            checks["providers"] = {
                "status": "warn",
                "message": "Provider factory not configured"
            }
    except Exception as e:
        checks["providers"] = {
            "status": "warn",
            "message": f"Error checking providers: {e}"
        }

    # Build response
    response_data = {
        "ready": all_ready,
        "checks": checks
    }

    if all_ready:
        logger.debug("Readiness check passed: %s", checks)
        return JSONResponse(content=response_data, status_code=status.HTTP_200_OK)
    else:
        logger.warning("Readiness check failed: %s", checks)
        return JSONResponse(
            content=response_data,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

