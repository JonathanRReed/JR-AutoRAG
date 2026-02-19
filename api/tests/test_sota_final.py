"""Tests for final SOTA enhancements."""

from app.core.budget_planner import BudgetClass, BudgetPlanner
from app.core.conflict_detector import ConflictDetector
from app.core.gatherer import EvidenceChunk


class TestConflictDetector:
    def test_no_conflicts(self):
        detector = ConflictDetector()
        chunks = [
            EvidenceChunk(id="1", title="A", snippet="Python is popular.", score=0.9),
            EvidenceChunk(id="2", title="B", snippet="Python runs on many platforms.", score=0.8),
        ]
        result = detector.detect(chunks)
        assert not result.has_conflicts

    def test_detects_negation_conflict(self):
        detector = ConflictDetector()
        chunks = [
            EvidenceChunk(id="1", title="A", snippet="Python is easy to learn. It is beginner-friendly.", score=0.9),
            EvidenceChunk(id="2", title="B", snippet="Python is not easy to learn. It has complex syntax.", score=0.8),
        ]
        result = detector.detect(chunks)
        assert result.has_conflicts
        assert result.resolution_strategy in ["prefer_higher_score", "multi_view_answer"]

class TestBudgetPlanner:
    def test_minimal_budget(self):
        planner = BudgetPlanner()
        plan = planner.plan(BudgetClass.MINIMAL)
        # Minimal budget constrains max iterations
        assert plan.max_iterations <= 2
        assert not plan.use_colbert
        assert not plan.use_graph

    def test_premium_budget(self):
        planner = BudgetPlanner()
        plan = planner.plan(BudgetClass.PREMIUM)
        assert plan.max_iterations > 1
        assert plan.use_raptor

    def test_complexity_scaling(self):
        planner = BudgetPlanner()
        simple = planner.plan(BudgetClass.STANDARD, query_complexity=0.2)
        complex_q = planner.plan(BudgetClass.STANDARD, query_complexity=0.9)
        assert complex_q.suggested_k >= simple.suggested_k
