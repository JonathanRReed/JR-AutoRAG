"""Conflict detection in retrieved evidence.

Identifies when retrieved chunks contain contradictory information
and provides resolution strategies.

This module implements conflict detection for SOTA RAG:
- Detects negation-based conflicts
- Detects numeric contradictions
- Provides resolution strategies for generation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk


@dataclass
class ConflictResult:
    """Result of conflict detection."""
    has_conflicts: bool
    conflicts: list[tuple[str, str, str]]  # (claim1, claim2, type)
    resolution_strategy: str
    consensus_claims: list[str]
    details: dict = field(default_factory=dict)


class ConflictDetector:
    """Detect and resolve conflicting evidence in retrieved chunks.

    Key features:
    - Detects negation conflicts (is vs is not)
    - Detects numeric contradictions (different values for same topic)
    - Suggests resolution strategies
    - Extracts consensus claims for reliable generation
    """

    # Negation patterns that indicate potential conflict
    NEGATION_PATTERNS = [
        (r'\bis\b', r'\bis not\b'),
        (r'\bis\b', r'\bisn\'t\b'),
        (r'\bcan\b', r'\bcannot\b'),
        (r'\bcan\b', r'\bcan\'t\b'),
        (r'\bwill\b', r'\bwill not\b'),
        (r'\bwill\b', r'\bwon\'t\b'),
        (r'\bshould\b', r'\bshould not\b'),
        (r'\bshould\b', r'\bshouldn\'t\b'),
        (r'\balways\b', r'\bnever\b'),
        (r'\bincreased\b', r'\bdecreased\b'),
        (r'\bhigher\b', r'\blower\b'),
        (r'\bmore\b', r'\bless\b'),
        (r'\bbetter\b', r'\bworse\b'),
        (r'\bsupports\b', r'\bopposes\b'),
        (r'\bconfirms\b', r'\bdenies\b'),
        (r'\btrue\b', r'\bfalse\b'),
        (r'\bcorrect\b', r'\bincorrect\b'),
        (r'\bvalid\b', r'\binvalid\b'),
    ]

    # Stopwords to exclude from overlap calculation
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'and', 'but', 'or',
        'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
    }

    def __init__(self, min_overlap: float = 0.4):
        """Initialize conflict detector.

        Args:
            min_overlap: Minimum term overlap to consider claims related
        """
        self.min_overlap = min_overlap
        self._compiled_patterns = [
            (re.compile(pos, re.IGNORECASE), re.compile(neg, re.IGNORECASE))
            for pos, neg in self.NEGATION_PATTERNS
        ]

    def _tokenize(self, text: str) -> set[str]:
        """Extract content words, filtering stopwords."""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return {w for w in words if w not in self.STOPWORDS}

    def _extract_claims(self, chunk: EvidenceChunk) -> list[str]:
        """Extract sentence-level claims from chunk."""
        sentences = re.split(r'(?<=[.!?])\s+', chunk.snippet)
        claims = []
        for s in sentences:
            s = s.strip()
            # Filter out very short sentences and headers
            if len(s.split()) > 3 and not s.startswith(('#', '**', '-', '*')):
                claims.append(s)
        return claims

    def _claims_conflict(self, claim1: str, claim2: str) -> tuple[bool, str]:
        """Check if two claims conflict.

        Returns:
            Tuple of (is_conflict, conflict_type)
        """
        # First check if claims are about the same topic
        terms1 = self._tokenize(claim1)
        terms2 = self._tokenize(claim2)

        if not terms1 or not terms2:
            return False, ""

        overlap = len(terms1 & terms2) / min(len(terms1), len(terms2))

        if overlap < self.min_overlap:
            # Claims are about different topics
            return False, ""

        # Check for negation pattern conflicts
        for pos_pattern, neg_pattern in self._compiled_patterns:
            pos_in_1 = pos_pattern.search(claim1)
            neg_in_1 = neg_pattern.search(claim1)
            pos_in_2 = pos_pattern.search(claim2)
            neg_in_2 = neg_pattern.search(claim2)

            # One claim has positive, other has negative
            if (pos_in_1 and neg_in_2) or (neg_in_1 and pos_in_2):
                return True, "negation"

        # Check for numeric contradictions
        nums1 = re.findall(r'\b(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand)?\b', claim1)
        nums2 = re.findall(r'\b(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand)?\b', claim2)

        if nums1 and nums2:
            # Extract numeric values
            val1 = float(nums1[0][0])
            val2 = float(nums2[0][0])

            # Same topic but significantly different numbers
            if overlap > 0.5 and abs(val1 - val2) / max(val1, val2, 1) > 0.2:
                return True, "numeric"

        return False, ""

    def detect(
        self,
        chunks: list[EvidenceChunk],
    ) -> ConflictResult:
        """Detect conflicts in retrieved evidence.

        Args:
            chunks: List of evidence chunks to analyze

        Returns:
            ConflictResult with detected conflicts and resolution strategy
        """
        # Extract all claims with their source
        all_claims: list[tuple[str, str, float]] = []  # (claim, chunk_id, score)
        for chunk in chunks:
            for claim in self._extract_claims(chunk):
                all_claims.append((claim, chunk.id, chunk.score))

        # Compare claims pairwise
        conflicts: list[tuple[str, str, str]] = []
        conflict_sources: dict[str, float] = {}  # chunk_id -> score for conflict resolution

        for i, (claim1, id1, score1) in enumerate(all_claims):
            for _j, (claim2, id2, score2) in enumerate(all_claims[i+1:], i+1):
                # Skip claims from same chunk
                if id1 == id2:
                    continue

                is_conflict, conflict_type = self._claims_conflict(claim1, claim2)
                if is_conflict:
                    conflicts.append((claim1, claim2, conflict_type))
                    conflict_sources[id1] = max(conflict_sources.get(id1, 0), score1)
                    conflict_sources[id2] = max(conflict_sources.get(id2, 0), score2)

        # Determine resolution strategy
        if not conflicts:
            strategy = "none_needed"
        elif len(conflicts) == 1:
            strategy = "prefer_higher_score"
        elif len({c[2] for c in conflicts}) == 1:
            # All conflicts are same type
            strategy = "prefer_higher_score"
        else:
            strategy = "multi_view_answer"

        # Extract consensus claims (non-conflicting)
        conflicting_claims = set()
        for c1, c2, _ in conflicts:
            conflicting_claims.add(c1)
            conflicting_claims.add(c2)
        consensus = [c for c, _, _ in all_claims if c not in conflicting_claims]

        # Build details
        details = {
            "total_claims_analyzed": len(all_claims),
            "conflict_types": list({c[2] for c in conflicts}),
            "conflicting_sources": list(conflict_sources.keys()),
        }

        return ConflictResult(
            has_conflicts=len(conflicts) > 0,
            conflicts=conflicts[:5],  # Limit to top 5
            resolution_strategy=strategy,
            consensus_claims=consensus[:10],
            details=details,
        )


__all__ = ["ConflictResult", "ConflictDetector"]
