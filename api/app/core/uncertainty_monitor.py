"""Confidence/uncertainty monitoring during generation."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

UNCERTAINTY_PATTERNS: tuple[str, ...] = (
    r"\b(maybe|perhaps|possibly|might|could be|I think|I believe)\b",
    r"\b(not sure|uncertain|unclear|unknown)\b",
    r"\b(approximately|around|about|roughly)\b",
    r"\b(appears to|seems to|it seems)\b",
    r"\?+$",
)

CONFIDENCE_PATTERNS: tuple[str, ...] = (
    r"\b(definitely|certainly|clearly|obviously|according to)\b",
    r"\b(research shows|studies indicate|data shows)\b",
    r"\[[0-9]+\]",
    r"\b(confirms|demonstrates|establishes)\b",
)


@dataclass
class ConfidenceSignal:
    """Snapshot of generation confidence for a span of text."""

    text: str
    heuristic_score: float
    avg_logprob: float | None = None
    entropy: float | None = None
    logit_margin: float | None = None
    aggregate: float = 0.0

    def should_trigger(self, threshold: float) -> bool:
        return self.aggregate < threshold


class UncertaintyMonitor:
    """Combines heuristics and optional model signals to detect uncertainty."""

    def __init__(
        self,
        threshold: float = 0.35,
        heuristic_weight: float = 0.6,
        uncertainty_patterns: Iterable[str] | None = None,
        confidence_patterns: Iterable[str] | None = None,
    ) -> None:
        self.threshold = threshold
        self.heuristic_weight = heuristic_weight
        unc_patterns = uncertainty_patterns or UNCERTAINTY_PATTERNS
        conf_patterns = confidence_patterns or CONFIDENCE_PATTERNS
        self._uncertainty_re = [re.compile(p, re.IGNORECASE) for p in unc_patterns]
        self._confidence_re = [re.compile(p, re.IGNORECASE) for p in conf_patterns]

    def score_heuristics(self, text: str) -> float:
        """Return heuristic confidence score in [0, 1]."""
        clean = text.strip()
        if not clean:
            return 0.4
        uncertainty_hits = sum(
            1 for pattern in self._uncertainty_re if pattern.search(clean)
        )
        confidence_hits = sum(
            1 for pattern in self._confidence_re if pattern.search(clean)
        )
        base = 0.55
        penalty = 0.12 * uncertainty_hits
        bonus = 0.15 * confidence_hits
        if len(clean) < 40:
            penalty += 0.08
        if len(clean) > 320:
            penalty += 0.05
        score = base - penalty + bonus
        return max(0.0, min(1.0, score))

    def _score_logprob(self, avg_logprob: float | None) -> float | None:
        if avg_logprob is None:
            return None
        try:
            return 1.0 / (1.0 + math.exp(-avg_logprob))
        except OverflowError:
            return 0.0 if avg_logprob < 0 else 1.0

    def _score_entropy(self, entropy: float | None) -> float | None:
        if entropy is None:
            return None
        # Assume entropy roughly in [0, 10]; convert to confidence
        entropy = max(0.0, min(10.0, entropy))
        return 1.0 - (entropy / 10.0)

    def _score_logit_margin(self, margin: float | None) -> float | None:
        if margin is None:
            return None
        # Margin already proportional to confidence; squash to [0,1]
        return max(0.0, min(1.0, 0.5 + (margin / 10.0)))

    def estimate(
        self,
        text: str,
        *,
        avg_logprob: float | None = None,
        entropy: float | None = None,
        logit_margin: float | None = None,
    ) -> ConfidenceSignal:
        """Combine available signals into a single confidence estimate."""
        heuristic_score = self.score_heuristics(text)
        extras: list[float] = []
        logprob_score = self._score_logprob(avg_logprob)
        if logprob_score is not None:
            extras.append(logprob_score)
        entropy_score = self._score_entropy(entropy)
        if entropy_score is not None:
            extras.append(entropy_score)
        margin_score = self._score_logit_margin(logit_margin)
        if margin_score is not None:
            extras.append(margin_score)

        if extras:
            stats_score = sum(extras) / len(extras)
            aggregate = (self.heuristic_weight * heuristic_score) + (
                (1 - self.heuristic_weight) * stats_score
            )
        else:
            aggregate = heuristic_score

        aggregate = max(0.0, min(1.0, aggregate))
        return ConfidenceSignal(
            text=text,
            heuristic_score=heuristic_score,
            avg_logprob=avg_logprob,
            entropy=entropy,
            logit_margin=logit_margin,
            aggregate=aggregate,
        )

    def should_trigger(self, signal: ConfidenceSignal) -> bool:
        return signal.aggregate < self.threshold


__all__ = [
    "CONFIDENCE_PATTERNS",
    "UNCERTAINTY_PATTERNS",
    "ConfidenceSignal",
    "UncertaintyMonitor",
]
