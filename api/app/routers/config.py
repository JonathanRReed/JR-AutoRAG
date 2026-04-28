from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..core.auth import get_auth
from ..core.document_acl import get_acl_enforcer, resolve_acl_defaults
from ..core.providers import ProviderError, discover_models
from ..schemas.config import (
    RETRIEVAL_PRESETS,
    AppConfig,
    ModelDownloadRequest,
    ModelStatusRequest,
    ModelStatusResponse,
    ProviderConfig,
    RetrievalDefaults,
)
from ..services import ServiceContainer, get_container

router = APIRouter()


def _check_model_cached(model_id: str) -> tuple[str, str | None]:
    try:
        from huggingface_hub import scan_cache_dir, snapshot_download
    except Exception as exc:
        return "unknown", f"Cache check unavailable: {exc}"
    try:
        snapshot_download(repo_id=model_id, local_files_only=True)
        return "installed", None
    except Exception as exc:
        try:
            cache_info = scan_cache_dir()
            if any(repo.repo_id == model_id for repo in cache_info.repos):
                return "installed", None
        except Exception:
            pass
        return "missing", str(exc)


def _download_model(model_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Download unavailable: {exc}") from exc
    try:
        snapshot_download(repo_id=model_id, local_files_only=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Download failed: {exc}") from exc


def _delete_model_cache(model_id: str) -> None:
    try:
        from huggingface_hub import scan_cache_dir
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete unavailable: {exc}") from exc
    try:
        cache_info = scan_cache_dir()
        cache_info.delete_repos(model_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc


@router.get("", response_model=AppConfig)
def read_config(container: ServiceContainer = Depends(get_container)):
    return container.config_store.read()


@router.put("", response_model=AppConfig)
def update_config(
    cfg: AppConfig,
    container: ServiceContainer = Depends(get_container),
    active_profile: str | None = Query(default=None, description="Optional provider profile to activate"),
):
    try:
        if active_profile and cfg.provider_profiles:
            profile = next((p for p in cfg.provider_profiles if p.name == active_profile), None)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Profile '{active_profile}' not found")
            cfg.provider = profile.provider
        sanitized = container.prepare_config_for_storage(cfg)
        stored = container.config_store.write(sanitized)
        container.apply_config(stored)
        return stored
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/models", response_model=list[str])
async def list_models(payload: ProviderConfig):
    try:
        models = await discover_models(payload)
        return models
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/models/status", response_model=ModelStatusResponse)
def model_status(payload: ModelStatusRequest, container: ServiceContainer = Depends(get_container)):
    embedding_status = "unknown"
    reranker_status = "unknown"
    embedding_message = None
    reranker_message = None
    cfg = container.config_store.read()

    if payload.embedding_model:
        embedding_status, embedding_message = _check_model_cached(payload.embedding_model)
    if payload.reranker_model:
        reranker_status, reranker_message = _check_model_cached(payload.reranker_model)

    return ModelStatusResponse(
        embedding=embedding_status,
        reranker=reranker_status,
        embedding_message=embedding_message,
        reranker_message=reranker_message,
        deployment_profile=cfg.deployment_profile,
        local_only_ready=cfg.deployment_profile != "local_only"
        or (
            cfg.provider is None
            or str(cfg.provider.base_url).startswith(("http://localhost", "http://127.0.0.1", "http://0.0.0.0"))
        ),
    )


@router.get("/policy")
def local_first_policy(container: ServiceContainer = Depends(get_container)):
    return container.local_first.describe()


@router.get("/recommendations")
def quality_recommendations(
    request: Request,
    container: ServiceContainer = Depends(get_container),
):
    cfg = container.config_store.read()
    docs = container.document_store.list()
    auth_enabled = get_auth().require_auth()
    scopes = getattr(request.state, "scopes", [])
    if auth_enabled and "admin" not in scopes:
        default_public, _ = resolve_acl_defaults(auth_enabled)
        enforcer = get_acl_enforcer(default_public=default_public)
        user_id = getattr(request.state, "user_id", None)
        docs = [
            doc
            for doc in docs
            if enforcer.check_access(doc.id, user_id, "read")[0]
        ]
    parser_counts: dict[str, int] = {}
    low_confidence = 0
    processing_errors = 0
    for doc in docs:
        parser = doc.metadata.get("parser_provider", "native")
        parser_counts[parser] = parser_counts.get(parser, 0) + 1
        try:
            confidence = float(doc.metadata.get("parser_confidence", "1") or 1)
        except ValueError:
            confidence = 1.0
        if confidence < 0.7:
            low_confidence += 1
        if doc.metadata.get("processing_status") == "error":
            processing_errors += 1

    retrieval = cfg.retrieval
    recommendations: list[dict[str, object]] = []
    if not docs:
        recommendations.append({
            "id": "ingest-corpus",
            "title": "Ingest a mixed benchmark corpus",
            "priority": "high",
            "detail": "Add PDFs, DOCX files, Markdown, and scanned material before promoting retrieval presets.",
            "action": "documents",
        })
    if parser_counts.get("docling", 0) == 0:
        recommendations.append({
            "id": "enable-docling",
            "title": "Install Docling for structured parsing",
            "priority": "high",
            "detail": "The parser adapter is ready and will prefer Docling when the package is installed locally.",
            "action": "parser",
        })
    if low_confidence:
        recommendations.append({
            "id": "review-low-confidence",
            "title": "Review low-confidence extractions",
            "priority": "medium",
            "detail": f"{low_confidence} document(s) have parser confidence below 0.70.",
            "action": "preview",
        })
    if not getattr(retrieval, "use_reranking", False):
        recommendations.append({
            "id": "enable-reranker",
            "title": "Enable reranking for expert answers",
            "priority": "medium",
            "detail": "Reranking usually improves context precision when enough local compute is available.",
            "action": "config",
        })
    if not getattr(retrieval, "enforce_evidence_contract", False):
        recommendations.append({
            "id": "evidence-contract",
            "title": "Turn on evidence contract checks",
            "priority": "medium",
            "detail": "Evidence contracts make answer generation stricter and easier to audit.",
            "action": "config",
        })
    if processing_errors:
        recommendations.append({
            "id": "processing-errors",
            "title": "Fix ingestion errors",
            "priority": "high",
            "detail": f"{processing_errors} document(s) failed background indexing.",
            "action": "documents",
        })

    return {
        "deployment_profile": cfg.deployment_profile,
        "document_count": len(docs),
        "parser_counts": parser_counts,
        "low_confidence_documents": low_confidence,
        "processing_errors": processing_errors,
        "active_features": {
            "reranking": getattr(retrieval, "use_reranking", False),
            "graph": getattr(retrieval, "graph", False),
            "raptor": getattr(retrieval, "raptor", False),
            "evidence_contract": getattr(retrieval, "enforce_evidence_contract", False),
            "local_only": cfg.deployment_profile.value == "local_only",
        },
        "recommendations": recommendations,
    }


@router.post("/models/download")
def download_model(payload: ModelDownloadRequest):
    kind = payload.kind.lower().strip()
    if kind not in {"embedding", "reranker"}:
        raise HTTPException(status_code=400, detail="kind must be 'embedding' or 'reranker'")
    if not payload.model:
        raise HTTPException(status_code=400, detail="model is required")
    _download_model(payload.model)
    return {"status": "ok", "model": payload.model}


@router.post("/models/delete")
def delete_model(payload: ModelDownloadRequest):
    kind = payload.kind.lower().strip()
    if kind not in {"embedding", "reranker"}:
        raise HTTPException(status_code=400, detail="kind must be 'embedding' or 'reranker'")
    if not payload.model:
        raise HTTPException(status_code=400, detail="model is required")
    _delete_model_cache(payload.model)
    return {"status": "ok", "model": payload.model}


@router.get("/presets", response_model=dict[str, RetrievalDefaults])
def list_presets():
    """List available retrieval presets (turbo, fast, balanced, thorough, ultra_accurate)."""
    return RETRIEVAL_PRESETS


@router.get("/presets/active")
def get_active_preset(container: ServiceContainer = Depends(get_container)):
    """Determine which preset the current config most closely matches."""
    cfg = container.config_store.read()
    current = cfg.retrieval

    # Find best matching preset by comparing key parameters
    best_match = "balanced"
    best_score = 0

    for name, preset in RETRIEVAL_PRESETS.items():
        score = 0
        # Compare key parameters
        if current.dense_k == preset.dense_k:
            score += 2
        elif abs(current.dense_k - preset.dense_k) <= 2:
            score += 1
        if current.use_reranking == preset.use_reranking:
            score += 1
        if current.raptor == preset.raptor:
            score += 1
        if current.graph == preset.graph:
            score += 1
        if current.flare_generation == preset.flare_generation:
            score += 1
        if current.enforce_evidence_contract == preset.enforce_evidence_contract:
            score += 1

        if score > best_score:
            best_score = score
            best_match = name

    return {
        "level": best_match,
        "features": {
            "reranking": current.use_reranking,
            "raptor": current.raptor,
            "graph": current.graph,
            "flare": current.flare_generation,
            "evidence_contract": current.enforce_evidence_contract,
        }
    }


@router.post("/presets/{preset_name}", response_model=AppConfig)
def apply_preset(
    preset_name: str,
    container: ServiceContainer = Depends(get_container),
):
    """Apply a retrieval preset to the current configuration."""
    preset_name_lower = preset_name.lower()
    if preset_name_lower not in RETRIEVAL_PRESETS:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{preset_name}' not found. Available: {list(RETRIEVAL_PRESETS.keys())}"
        )

    cfg = container.config_store.read()
    cfg.retrieval = RETRIEVAL_PRESETS[preset_name_lower].model_copy()
    stored = container.config_store.write(cfg)
    container.apply_config(stored)
    return stored
