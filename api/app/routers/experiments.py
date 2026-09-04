"""Local AutoRAG experiment endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.auth import get_auth
from ..core.experiments import (
    ExperimentConfig as CoreExperimentConfig,
    ExperimentRunStore,
    LocalExperimentRunner,
    apply_winning_preset,
)
from ..schemas.experiments import (
    ExperimentConfig,
    ExperimentPromoteResponse,
    ExperimentRun,
)
from ..services import ServiceContainer, get_container

router = APIRouter(prefix="/experiments", tags=["experiments"])


def get_experiment_store(
    container: ServiceContainer = Depends(get_container),
) -> ExperimentRunStore:
    data_dir = Path(container.config_store.path).parent
    return ExperimentRunStore(data_dir / "experiments.json")


def _experiment_owner_filter(request: Request) -> str | None:
    if not get_auth().require_auth():
        return None
    scopes = getattr(request.state, "scopes", [])
    if "admin" in scopes:
        return None
    return str(getattr(request.state, "user_id", "") or "anonymous")


def _experiment_owner_for_create(request: Request) -> str:
    if not get_auth().require_auth():
        return "anonymous"
    return str(getattr(request.state, "user_id", "") or "anonymous")


@router.get("", response_model=list[ExperimentRun])
def list_experiments(
    request: Request,
    limit: int = 50,
    store: ExperimentRunStore = Depends(get_experiment_store),
):
    return store.list(
        limit=max(1, min(limit, 200)), owner_id=_experiment_owner_filter(request)
    )


@router.post("", response_model=ExperimentRun)
def run_experiment(
    payload: ExperimentConfig,
    request: Request,
    container: ServiceContainer = Depends(get_container),
    store: ExperimentRunStore = Depends(get_experiment_store),
):
    cfg = CoreExperimentConfig(**payload.model_dump())
    runner = LocalExperimentRunner(store)
    return runner.run(
        cfg=cfg,
        app_config=container.config_store.read(),
        doc_count=len(container.document_store.list()),
        owner_id=_experiment_owner_for_create(request),
    )


@router.get("/{experiment_id}", response_model=ExperimentRun)
def get_experiment(
    experiment_id: str,
    request: Request,
    store: ExperimentRunStore = Depends(get_experiment_store),
):
    run = store.get(experiment_id, owner_id=_experiment_owner_filter(request))
    if run is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return run


@router.post("/{experiment_id}/promote", response_model=ExperimentPromoteResponse)
def promote_experiment(
    experiment_id: str,
    request: Request,
    container: ServiceContainer = Depends(get_container),
    store: ExperimentRunStore = Depends(get_experiment_store),
):
    run = store.get(experiment_id, owner_id=_experiment_owner_filter(request))
    if run is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    preset_name = str(run.get("winning_preset") or "balanced")
    cfg = apply_winning_preset(container.config_store.read(), preset_name)
    stored = container.config_store.write(cfg)
    container.apply_config(stored)
    promoted = store.mark_promoted(experiment_id) or run
    return ExperimentPromoteResponse(run=promoted, promoted_preset=preset_name)
