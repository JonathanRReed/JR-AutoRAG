"""Router for detailed artifact data (G4)."""

from fastapi import APIRouter, HTTPException

from ..state import get_orchestrator

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("/graph")
async def get_graph_data() -> dict:
    """Get detailed GraphRAG entity and community data."""
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return orchestrator.get_graph_data()


@router.get("/raptor")
async def get_raptor_data() -> dict:
    """Get detailed RAPTOR tree data."""
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return orchestrator.get_raptor_data()


@router.post("/build")
async def trigger_build(force: bool = False) -> dict:
    """Manually trigger background artifact build (G4)."""
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return orchestrator.trigger_artifact_build(force=force)
