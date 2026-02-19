"""Cache management API routes."""

from fastapi import APIRouter, Depends

from ..core.artifact_builder import get_artifact_builder
from ..core.cache import get_cache_manager
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/api/cache", tags=["cache"])


@router.delete("/clear", status_code=200)
async def clear_cache(
    include_disk: bool = True,
    container: ServiceContainer = Depends(get_container),
):
    """Clear all cached indexes and query results.

    Args:
        include_disk: If True, also delete persisted index files on disk.

    Returns:
        Status and confirmation message.
    """
    cache_manager = get_cache_manager()

    # Clear in-memory caches
    cache_manager.embeddings.clear()
    cache_manager.queries.invalidate_all()

    # Reset artifact builder state (G4)
    get_artifact_builder().reset()

    # Clear retrieval engine cache (both in-memory and optionally disk)
    container.retrieval_engine.clear_cache(include_disk=include_disk)

    return {
        "status": "cleared",
        "message": "All caches invalidated",
        "disk_cleared": include_disk,
    }


@router.get("/status")
async def cache_status(container: ServiceContainer = Depends(get_container)):
    """Get current cache statistics.

    Returns:
        Cache statistics including embedding cache, query cache, and index status.
    """
    cache_manager = get_cache_manager()

    index_valid = bool(container.retrieval_engine._chunks)
    chunk_count = len(container.retrieval_engine._chunks) if index_valid else 0

    return {
        "embeddings": cache_manager.embeddings.stats(),
        "queries": cache_manager.queries.stats(),
        "index": {
            "valid": index_valid,
            "chunk_count": chunk_count,
            "corpus_version": container.retrieval_engine.get_corpus_version(),
        },
    }


@router.post("/rebuild", status_code=200)
async def rebuild_index(container: ServiceContainer = Depends(get_container)):
    """Force a full index rebuild.

    This will clear existing index and rebuild from all documents.

    Returns:
        Status and new index statistics.
    """
    # Clear and rebuild
    container.retrieval_engine.clear_cache(include_disk=True)
    container.retrieval_engine.build()

    return {
        "status": "rebuilt",
        "chunk_count": len(container.retrieval_engine._chunks),
        "corpus_version": container.retrieval_engine.get_corpus_version(),
    }
