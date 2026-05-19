from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.telemetry import PipelineStep, TelemetryStore


def test_metrics_cache_hit_rate_uses_embedding_hits(tmp_path):
    store = TelemetryStore(tmp_path / "traces.json")
    store.record(
        prompt="q1",
        answer="a1",
        metrics={
            "duration_ms": 100,
            "context_chunks": 4,
            "embedding_cache_hits": 3,
            "embedding_cache_misses": 1,
            "quality_rating": "high",
        },
    )
    store.record(
        prompt="q2",
        answer="a2",
        metrics={
            "duration_ms": 200,
            "context_chunks": 2,
            "embedding_cache_hits": 1,
            "embedding_cache_misses": 1,
            "quality_rating": "low",
        },
    )

    metrics = store.export_metrics()
    assert metrics["cache_hit_rate"] == pytest.approx(4 / 6)
    assert metrics["avg_chunks_per_query"] == pytest.approx(3.0)


def test_metrics_distribution_and_rerank_usage(tmp_path):
    store = TelemetryStore(tmp_path / "traces.json")
    now = datetime.now(UTC)
    steps = [
        PipelineStep(
            name="retrieval",
            started_at=now,
            completed_at=now,
            duration_ms=1.0,
            details={"reranked": True},
        )
    ]
    store.record(
        prompt="q",
        answer="a",
        metrics={
            "duration_ms": 10,
            "context_chunks": 1,
            "retrieval_mode": "graph",
            "quality_rating": "insufficient",
            "flare_retrievals": 2,
        },
        steps=steps,
    )

    dist = store.get_retrieval_mode_distribution()
    quality = store._calculate_quality_distribution()
    assert dist["graph"] == 1
    assert quality["insufficient"] == 1
    assert store._calculate_rerank_usage() == pytest.approx(1.0)
    assert store.get_flare_trigger_rate() == pytest.approx(1.0)
