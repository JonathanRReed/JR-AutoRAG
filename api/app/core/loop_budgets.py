"""Loop budgets for iterative RAG strategies.

Provides comprehensive budget enforcement for iterative loops like:
- Self-RAG iterative refinement
- FLARE predictive retrieval
- CRAG corrective retrieval
- Multi-hop reasoning loops

Features:
- Max iteration limits
- Token consumption tracking
- Confidence-based early stopping
- Evidence stability detection
- Wall-clock timeouts
- Automatic fallback to simpler plans
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
import logging

logger = logging.getLogger("autorag.loop_budgets")


# =============================================================================
# Stop Reasons
# =============================================================================

class StopReason(Enum):
    """Reason why a loop stopped."""
    COMPLETE = "complete"  # Natural completion
    MAX_ITERATIONS = "max_iterations"
    MAX_RETRIEVALS = "max_retrievals"
    MAX_RERANKS = "max_reranks"
    MAX_TOKENS = "max_tokens"
    MAX_WALL_CLOCK = "max_wall_clock"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    EVIDENCE_STABLE = "evidence_stable"
    ERROR = "error"
    FALLBACK = "fallback"  # Fell back to simpler strategy


# =============================================================================
# Loop Budget Configuration
# =============================================================================

@dataclass
class LoopBudget:
    """Budget configuration for iterative loops.
    
    All limits are checked on each iteration, and the loop
    stops when any limit is reached.
    """
    # Iteration limits
    max_iterations: int = 3
    max_added_retrievals: int = 10  # New docs retrieved across all iterations
    max_reranks: int = 5
    
    # Token limits
    max_input_tokens: int = 8000
    max_output_tokens: int = 2000
    max_total_tokens: int = 12000
    
    # Time limits
    max_wall_clock_seconds: float = 30.0
    
    # Early stopping
    confidence_early_stop: float = 0.95  # Stop if confidence exceeds this
    evidence_stability_threshold: int = 2  # Stop if top-k evidence stable for N iterations
    
    # Fallback behavior
    allow_fallback: bool = True  # Fall back to simple plan on budget exhaustion
    fallback_after_iterations: int = 2  # Force fallback after this many iterations
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_added_retrievals": self.max_added_retrievals,
            "max_reranks": self.max_reranks,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_wall_clock_seconds": self.max_wall_clock_seconds,
            "confidence_early_stop": self.confidence_early_stop,
            "evidence_stability_threshold": self.evidence_stability_threshold,
            "allow_fallback": self.allow_fallback,
        }
    
    @classmethod
    def strict(cls) -> "LoopBudget":
        """Get strict budget for high-stakes applications."""
        return cls(
            max_iterations=2,
            max_added_retrievals=5,
            max_reranks=2,
            max_total_tokens=6000,
            max_wall_clock_seconds=15.0,
            confidence_early_stop=0.90,
        )
    
    @classmethod
    def relaxed(cls) -> "LoopBudget":
        """Get relaxed budget for research/exploration."""
        return cls(
            max_iterations=5,
            max_added_retrievals=25,
            max_reranks=10,
            max_total_tokens=20000,
            max_wall_clock_seconds=120.0,
            confidence_early_stop=0.99,
        )


# =============================================================================
# Loop State
# =============================================================================

@dataclass
class LoopState:
    """Current state of an iterative loop."""
    # Counters
    iteration: int = 0
    total_retrievals: int = 0
    total_reranks: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # Timing
    start_time: float = field(default_factory=time.time)
    
    # Evidence tracking
    evidence_history: list[list[str]] = field(default_factory=list)  # List of top-k doc IDs per iteration
    
    # Confidence tracking
    confidence_history: list[float] = field(default_factory=list)
    
    @property
    def elapsed_seconds(self) -> float:
        """Time elapsed since loop started."""
        return time.time() - self.start_time
    
    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.total_input_tokens + self.total_output_tokens
    
    def record_iteration(
        self,
        retrievals: int = 0,
        reranks: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        top_evidence_ids: Optional[list[str]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Record metrics from an iteration."""
        self.iteration += 1
        self.total_retrievals += retrievals
        self.total_reranks += reranks
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        if top_evidence_ids is not None:
            self.evidence_history.append(top_evidence_ids)
        
        if confidence is not None:
            self.confidence_history.append(confidence)
    
    def evidence_is_stable(self, threshold: int) -> bool:
        """Check if evidence has been stable for threshold iterations."""
        if len(self.evidence_history) < threshold:
            return False
        
        # Compare last N evidence sets
        recent = self.evidence_history[-threshold:]
        first = set(recent[0])
        
        for evidence in recent[1:]:
            if set(evidence) != first:
                return False
        
        return True
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "total_retrievals": self.total_retrievals,
            "total_reranks": self.total_reranks,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "stable_evidence": self.evidence_is_stable(2),
        }


# =============================================================================
# Stop Decision
# =============================================================================

@dataclass
class StopDecision:
    """Decision about whether to continue or stop a loop."""
    should_stop: bool
    reason: StopReason
    should_fallback: bool = False
    message: str = ""
    
    @classmethod
    def continue_loop(cls) -> "StopDecision":
        """Decision to continue the loop."""
        return cls(
            should_stop=False,
            reason=StopReason.COMPLETE,
            message="Continue iteration",
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "should_stop": self.should_stop,
            "reason": self.reason.value,
            "should_fallback": self.should_fallback,
            "message": self.message,
        }


# =============================================================================
# Budget Enforcer
# =============================================================================

class LoopBudgetEnforcer:
    """Enforces budgets on iterative loops.
    
    Usage:
        budget = LoopBudget(max_iterations=3)
        enforcer = LoopBudgetEnforcer(budget)
        state = LoopState()
        
        while True:
            decision = enforcer.check_and_decide(state)
            if decision.should_stop:
                if decision.should_fallback:
                    return simple_fallback()
                break
            
            # Do iteration work...
            state.record_iteration(retrievals=5, confidence=0.8)
    """
    
    def __init__(self, budget: LoopBudget) -> None:
        self.budget = budget
    
    def check_and_decide(self, state: LoopState) -> StopDecision:
        """Check all budgets and decide whether to stop.
        
        Returns StopDecision with reason and fallback recommendation.
        """
        # Check iteration limit
        if state.iteration >= self.budget.max_iterations:
            return StopDecision(
                should_stop=True,
                reason=StopReason.MAX_ITERATIONS,
                should_fallback=self.budget.allow_fallback,
                message=f"Reached max iterations ({self.budget.max_iterations})",
            )
        
        # Check retrieval limit
        if state.total_retrievals >= self.budget.max_added_retrievals:
            return StopDecision(
                should_stop=True,
                reason=StopReason.MAX_RETRIEVALS,
                should_fallback=self.budget.allow_fallback,
                message=f"Reached max retrievals ({self.budget.max_added_retrievals})",
            )
        
        # Check rerank limit
        if state.total_reranks >= self.budget.max_reranks:
            return StopDecision(
                should_stop=True,
                reason=StopReason.MAX_RERANKS,
                should_fallback=self.budget.allow_fallback,
                message=f"Reached max reranks ({self.budget.max_reranks})",
            )
        
        # Check token limits
        if state.total_tokens >= self.budget.max_total_tokens:
            return StopDecision(
                should_stop=True,
                reason=StopReason.MAX_TOKENS,
                should_fallback=self.budget.allow_fallback,
                message=f"Reached token budget ({self.budget.max_total_tokens})",
            )
        
        # Check wall clock
        if state.elapsed_seconds >= self.budget.max_wall_clock_seconds:
            return StopDecision(
                should_stop=True,
                reason=StopReason.MAX_WALL_CLOCK,
                should_fallback=self.budget.allow_fallback,
                message=f"Reached time limit ({self.budget.max_wall_clock_seconds}s)",
            )
        
        # Check confidence early stopping
        if state.confidence_history:
            latest_confidence = state.confidence_history[-1]
            if latest_confidence >= self.budget.confidence_early_stop:
                return StopDecision(
                    should_stop=True,
                    reason=StopReason.CONFIDENCE_THRESHOLD,
                    should_fallback=False,  # Good stop, no fallback needed
                    message=f"Confidence threshold met ({latest_confidence:.3f})",
                )
        
        # Check evidence stability
        if state.evidence_is_stable(self.budget.evidence_stability_threshold):
            return StopDecision(
                should_stop=True,
                reason=StopReason.EVIDENCE_STABLE,
                should_fallback=False,  # Good stop, evidence converged
                message=f"Evidence stable for {self.budget.evidence_stability_threshold} iterations",
            )
        
        # Check forced fallback
        if (self.budget.allow_fallback and 
            state.iteration >= self.budget.fallback_after_iterations):
            # Don't stop yet, but mark for potential fallback
            pass
        
        return StopDecision.continue_loop()
    
    def estimate_remaining_budget(self, state: LoopState) -> dict[str, Any]:
        """Estimate remaining budget for each dimension."""
        return {
            "iterations_remaining": max(0, self.budget.max_iterations - state.iteration),
            "retrievals_remaining": max(0, self.budget.max_added_retrievals - state.total_retrievals),
            "reranks_remaining": max(0, self.budget.max_reranks - state.total_reranks),
            "tokens_remaining": max(0, self.budget.max_total_tokens - state.total_tokens),
            "time_remaining_seconds": max(0, self.budget.max_wall_clock_seconds - state.elapsed_seconds),
        }
    
    def suggest_reduced_scope(self, state: LoopState) -> dict[str, Any]:
        """Suggest reduced scope based on remaining budget.
        
        Returns parameters that downstream stages should use to
        conserve remaining budget.
        """
        remaining = self.estimate_remaining_budget(state)
        
        suggestions = {}
        
        # Reduce k for retrieval if low on budget
        if remaining["retrievals_remaining"] < 5:
            suggestions["retrieval_k"] = min(3, remaining["retrievals_remaining"])
        
        # Reduce rerank pool if low on budget
        if remaining["reranks_remaining"] < 2:
            suggestions["rerank_pool_size"] = 10
        
        # Suggest shorter answers if low on tokens
        if remaining["tokens_remaining"] < 500:
            suggestions["max_answer_tokens"] = 200
            suggestions["concise_mode"] = True
        
        # Suggest skipping optional stages if low on time
        if remaining["time_remaining_seconds"] < 5:
            suggestions["skip_graphs"] = True
            suggestions["skip_compression"] = True
        
        return suggestions


# =============================================================================
# Presets
# =============================================================================

LOOP_BUDGETS = {
    "default": LoopBudget(),
    "strict": LoopBudget.strict(),
    "relaxed": LoopBudget.relaxed(),
    "single_shot": LoopBudget(max_iterations=1, allow_fallback=False),
    "deep_research": LoopBudget(
        max_iterations=5,
        max_added_retrievals=30,
        max_reranks=15,
        max_total_tokens=25000,
        max_wall_clock_seconds=180.0,
        confidence_early_stop=0.99,
    ),
}


def get_loop_budget(preset: str = "default") -> LoopBudget:
    """Get a loop budget by preset name."""
    return LOOP_BUDGETS.get(preset, LOOP_BUDGETS["default"])


__all__ = [
    "LoopBudget",
    "LoopState",
    "StopDecision",
    "StopReason",
    "LoopBudgetEnforcer",
    "LOOP_BUDGETS",
    "get_loop_budget",
]
