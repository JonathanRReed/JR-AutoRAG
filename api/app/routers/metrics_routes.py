"""Metrics API routes for 3.0 features."""


from fastapi import APIRouter

from ..core.corpus_health import get_corpus_health_checker
from ..core.model_roles import RECOMMENDED_MODELS, Role
from ..core.preset_metrics import get_preset_metrics_tracker
from ..core.retrieval_quality import get_retrieval_quality_gates

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/presets/estimates")
def get_preset_estimates():
    """Get latency/token estimates for all presets.

    Returns estimates based on last 20 runs per preset.
    """
    tracker = get_preset_metrics_tracker()
    estimates = tracker.get_all_estimates()

    return {
        "estimates": {
            preset: est.to_dict() for preset, est in estimates.items()
        },
        "message": f"Estimates for {len(estimates)} presets" if estimates else "No data yet",
    }


@router.get("/presets/estimates/{preset}")
def get_preset_estimate(preset: str):
    """Get latency/token estimate for a specific preset."""
    tracker = get_preset_metrics_tracker()
    estimate = tracker.get_estimate(preset)

    if estimate:
        return estimate.to_dict()
    return {"error": f"No data for preset '{preset}'", "sample_count": 0}


@router.get("/models/recommendations")
def get_model_recommendations():
    """Get recommended models per role and provider.

    Returns the RECOMMENDED_MODELS configuration.
    """
    return {
        "recommendations": {
            provider: {role.value: model for role, model in roles.items()}
            for provider, roles in RECOMMENDED_MODELS.items()
        },
        "roles": [r.value for r in Role],
    }


@router.delete("/presets/clear")
def clear_preset_metrics(preset: str | None = None):
    """Clear metrics data for a preset or all presets."""
    tracker = get_preset_metrics_tracker()
    tracker.clear(preset)

    return {
        "cleared": preset or "all",
        "message": f"Cleared metrics for {preset or 'all presets'}",
    }


@router.get("/corpus/health")
def get_corpus_health():
    """Get corpus health report with stats and checks.

    Returns overall health status, stats, and recommendations.
    """
    checker = get_corpus_health_checker()
    report = checker.generate_report()
    return report.to_dict()


@router.get("/corpus/stats")
def get_corpus_stats():
    """Get basic corpus statistics."""
    checker = get_corpus_health_checker()
    stats = checker.get_stats()
    return stats.to_dict()


@router.get("/quality/gates")
def get_quality_gates_config():
    """Get current retrieval quality gate configuration."""
    gates = get_retrieval_quality_gates()
    return {
        "min_similarity": gates.min_similarity,
        "min_evidence_count": gates.min_evidence_count,
        "high_risk_similarity": gates.high_risk_similarity,
        "critical_risk_count": gates.critical_risk_count,
    }

