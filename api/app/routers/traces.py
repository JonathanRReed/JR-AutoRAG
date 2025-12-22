"""Trace export and artifact build status router.

Implements E1: Trace bundle export endpoints for offline reproducibility.
Also provides artifact build status (G4).
"""

from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import JSONResponse

from ..state import get_orchestrator

router = APIRouter(tags=["traces"])


@router.get("/traces/download")
async def download_trace() -> Response:
    """Download the last query's trace bundle as JSON (E1).
    
    Returns:
        JSON file download with trace bundle
    """
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    trace_json = orchestrator.export_trace_json()
    if trace_json is None:
        raise HTTPException(
            status_code=404, 
            detail="No trace bundle available. Run a query first."
        )
    
    return Response(
        content=trace_json,
        media_type="application/json",
        headers={
            "Content-Disposition": "attachment; filename=trace_bundle.json",
        },
    )


@router.get("/traces/last")
async def get_last_trace() -> dict:
    """Get the last query's trace bundle as JSON object (E1).
    
    Returns:
        Trace bundle dict
    """
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    bundle = orchestrator.get_trace_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=404, 
            detail="No trace bundle available. Run a query first."
        )
    
    return bundle.to_dict()


@router.get("/artifacts/status")
async def get_artifact_status() -> dict:
    """Get current status of background artifact builds (G4).
    
    Returns:
        Dict with graph_rag and raptor build status
    """
    orchestrator = get_orchestrator()
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return orchestrator.get_artifact_build_status()
