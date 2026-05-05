"""Smoke tests for JR-AutoRAG 3.0 core components.
These tests ensure that the most critical 3.0 upgrade modules are functional.
"""

from app.core.context_metrics import ContextMetrics, GroundingInfo
from app.core.corpus_health import CorpusHealthChecker
from app.core.onboarding import create_onboarding_flow
from app.core.retrieval_quality import RetrievalQualityGates


def test_context_metrics_logic():
    """Verify P0.5: Context overflow logic works."""
    cm = ContextMetrics(tokens_used=5000, max_context_tokens=4096, chunks_used=10, chunks_dropped=2, tokens_dropped=500)
    assert cm.is_overflow is True
    assert "ratio" in cm.label.lower()
    assert cm.utilization_ratio > 1.0

def test_grounding_info_logic():
    """Verify P1.8: Grounding info logic works."""
    gi = GroundingInfo(is_grounded=True, docs_used=3, citations_kept=5, chunks_total=10, chunks_dropped=2)
    assert gi.grounding_label == "Grounded (3 docs, 5 citations)"

def test_retrieval_quality_gates():
    """Verify P2.14: Retrieval quality gates work."""
    gates = RetrievalQualityGates(min_similarity=0.7)
    chunks = [{'score': 0.8}, {'score': 0.9}]
    report = gates.evaluate(chunks)
    assert report.all_gates_passed is True
    assert report.hallucination_risk.value == "low"

def test_corpus_health_module():
    """Verify P2.15: Corpus health module can be initialized."""
    checker = CorpusHealthChecker()
    report = checker.generate_report()
    assert report.overall_status in ["healthy", "warning", "critical"]

def test_onboarding_flow_creation():
    """Verify P1.11: Onboarding flow logic."""
    flow = create_onboarding_flow()
    assert len(flow.steps) > 0
    assert flow.progress_percent == 0.0

    # Complete a step
    flow.complete_step(flow.steps[0].id)
    assert flow.progress_percent > 0.0
