"""Retrieval quality controls.

This module implements P2.14: Retrieval Quality Controls
- Min similarity threshold gate
- Min evidence count gate  
- Hallucination risk flag
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger("autorag.retrieval_quality")


class HallucinationRisk(str, Enum):
    """Risk level for hallucination based on evidence quality."""
    
    LOW = "low"           # Strong evidence, high similarity
    MEDIUM = "medium"     # Adequate evidence, decent similarity
    HIGH = "high"         # Weak evidence or low similarity
    CRITICAL = "critical" # No evidence or very poor quality


@dataclass
class QualityGateResult:
    """Result of quality gate checks."""
    
    passed: bool
    gate_name: str
    actual_value: float | int
    threshold: float | int
    message: str
    
    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "gate": self.gate_name,
            "value": self.actual_value,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class RetrievalQualityReport:
    """Complete quality assessment of retrieved evidence."""
    
    gates_passed: list[QualityGateResult]
    gates_failed: list[QualityGateResult]
    hallucination_risk: HallucinationRisk
    risk_factors: list[str]
    proceed_with_answer: bool
    warnings: list[str]
    
    @property
    def all_gates_passed(self) -> bool:
        return len(self.gates_failed) == 0
    
    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_gates_passed,
            "passed": [g.to_dict() for g in self.gates_passed],
            "failed": [g.to_dict() for g in self.gates_failed],
            "hallucination_risk": self.hallucination_risk.value,
            "risk_factors": self.risk_factors,
            "proceed": self.proceed_with_answer,
            "warnings": self.warnings,
        }


class RetrievalQualityGates:
    """Apply quality control gates to retrieved evidence.
    
    Gates:
    1. Min similarity threshold: chunks must meet min score
    2. Min evidence count: must have enough chunks
    3. Hallucination risk: flag low-evidence answers
    """
    
    def __init__(
        self,
        min_similarity: float = 0.3,
        min_evidence_count: int = 2,
        high_risk_similarity: float = 0.2,
        critical_risk_count: int = 0,
    ) -> None:
        """Initialize quality gates.
        
        Args:
            min_similarity: Minimum chunk similarity score (0-1)
            min_evidence_count: Minimum number of chunks needed
            high_risk_similarity: Similarity below which risk is high
            critical_risk_count: Evidence count at which risk is critical
        """
        self.min_similarity = min_similarity
        self.min_evidence_count = min_evidence_count
        self.high_risk_similarity = high_risk_similarity
        self.critical_risk_count = critical_risk_count
    
    def check_similarity_gate(
        self,
        chunks: list,
        key: str = "score",
    ) -> QualityGateResult:
        """Check if chunks meet minimum similarity threshold."""
        if not chunks:
            return QualityGateResult(
                passed=False,
                gate_name="min_similarity",
                actual_value=0.0,
                threshold=self.min_similarity,
                message="No chunks to evaluate",
            )
        
        # Get scores
        scores = []
        for c in chunks:
            score = getattr(c, key, None) or (c.get(key) if isinstance(c, dict) else 0)
            if score is not None:
                scores.append(float(score))
        
        if not scores:
            return QualityGateResult(
                passed=False,
                gate_name="min_similarity",
                actual_value=0.0,
                threshold=self.min_similarity,
                message="No scores found on chunks",
            )
        
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        
        # Pass if best chunk exceeds threshold
        passed = max_score >= self.min_similarity
        
        return QualityGateResult(
            passed=passed,
            gate_name="min_similarity",
            actual_value=round(max_score, 3),
            threshold=self.min_similarity,
            message=f"Best match: {max_score:.2f}, avg: {avg_score:.2f}" if passed 
                    else f"No chunks above {self.min_similarity} threshold (best: {max_score:.2f})",
        )
    
    def check_evidence_count_gate(
        self,
        chunks: list,
    ) -> QualityGateResult:
        """Check if enough evidence chunks are available."""
        count = len(chunks)
        passed = count >= self.min_evidence_count
        
        return QualityGateResult(
            passed=passed,
            gate_name="min_evidence_count",
            actual_value=count,
            threshold=self.min_evidence_count,
            message=f"{count} chunks retrieved" if passed 
                    else f"Only {count} chunks (need {self.min_evidence_count}+)",
        )
    
    def assess_hallucination_risk(
        self,
        chunks: list,
        avg_similarity: float,
    ) -> tuple[HallucinationRisk, list[str]]:
        """Assess risk of hallucination based on evidence quality.
        
        Returns:
            Tuple of (risk level, risk factors)
        """
        factors = []
        
        if len(chunks) <= self.critical_risk_count:
            factors.append("No supporting evidence")
            return HallucinationRisk.CRITICAL, factors
        
        if len(chunks) < self.min_evidence_count:
            factors.append(f"Low evidence count ({len(chunks)})")
        
        if avg_similarity < self.high_risk_similarity:
            factors.append(f"Very low similarity ({avg_similarity:.2f})")
        elif avg_similarity < self.min_similarity:
            factors.append(f"Below-threshold similarity ({avg_similarity:.2f})")
        
        # Determine risk level
        if len(factors) >= 2 or "No supporting evidence" in factors:
            risk = HallucinationRisk.CRITICAL
        elif "Very low similarity" in factors or len(chunks) == 0:
            risk = HallucinationRisk.HIGH
        elif factors:
            risk = HallucinationRisk.MEDIUM
        else:
            risk = HallucinationRisk.LOW
        
        return risk, factors
    
    def evaluate(
        self,
        chunks: list,
        score_key: str = "score",
    ) -> RetrievalQualityReport:
        """Run all quality gates and return complete report."""
        passed = []
        failed = []
        warnings = []
        
        # Gate 1: Similarity
        sim_gate = self.check_similarity_gate(chunks, score_key)
        if sim_gate.passed:
            passed.append(sim_gate)
        else:
            failed.append(sim_gate)
            warnings.append(f"Similarity below threshold: {sim_gate.message}")
        
        # Gate 2: Evidence count
        count_gate = self.check_evidence_count_gate(chunks)
        if count_gate.passed:
            passed.append(count_gate)
        else:
            failed.append(count_gate)
            warnings.append(f"Insufficient evidence: {count_gate.message}")
        
        # Compute average similarity for risk assessment
        scores = []
        for c in chunks:
            score = getattr(c, score_key, None) or (c.get(score_key) if isinstance(c, dict) else None)
            if score is not None:
                scores.append(float(score))
        avg_sim = sum(scores) / len(scores) if scores else 0.0
        
        # Hallucination risk
        risk, risk_factors = self.assess_hallucination_risk(chunks, avg_sim)
        
        if risk in (HallucinationRisk.HIGH, HallucinationRisk.CRITICAL):
            warnings.append(f"⚠️ {risk.value.upper()} hallucination risk")
        
        # Decision to proceed
        proceed = len(failed) == 0 or risk == HallucinationRisk.LOW
        
        return RetrievalQualityReport(
            gates_passed=passed,
            gates_failed=failed,
            hallucination_risk=risk,
            risk_factors=risk_factors,
            proceed_with_answer=proceed,
            warnings=warnings,
        )


# Global instance with defaults
_quality_gates: RetrievalQualityGates | None = None


def get_retrieval_quality_gates(
    min_similarity: float = 0.3,
    min_evidence_count: int = 2,
) -> RetrievalQualityGates:
    """Get or create quality gates with specified thresholds."""
    global _quality_gates
    if _quality_gates is None:
        _quality_gates = RetrievalQualityGates(
            min_similarity=min_similarity,
            min_evidence_count=min_evidence_count,
        )
    return _quality_gates


__all__ = [
    "HallucinationRisk",
    "QualityGateResult",
    "RetrievalQualityReport",
    "RetrievalQualityGates",
    "get_retrieval_quality_gates",
]
