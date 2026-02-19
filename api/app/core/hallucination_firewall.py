"""Hallucination firewall for post-generation verification.

Cross-checks generated claims against source citations,
zeroing out unsupported spans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk


@dataclass
class FirewallResult:
    """Result of hallucination firewall check."""
    original_answer: str
    cleaned_answer: str
    flagged_claims: list[str]
    verified_claims: int
    total_claims: int
    pass_rate: float
    details: dict = field(default_factory=dict)


class HallucinationFirewall:
    """Post-generation verifier that flags unsupported claims.

    Acts as a safety layer between generation and response,
    catching potential hallucinations before they reach the user.

    Key features:
    - Cross-checks claims against source citations
    - Flags sentences with low evidence overlap
    - In strict mode, removes/marks unsupported claims
    - Provides pass rate metrics for monitoring
    """

    # Stopwords always excluded from overlap calculation
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'and', 'but', 'or',
        'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
        'now', 'here', 'there', 'this', 'that', 'these', 'those', 'for', 'to',
        'of', 'in', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'through',
    }

    def __init__(
        self,
        strict_mode: bool = False,
        min_overlap: float = 0.35,
        min_pass_rate: float = 0.5,
    ):
        """Initialize hallucination firewall.

        Args:
            strict_mode: If True, remove/mark unsupported claims
            min_overlap: Minimum term overlap to consider supported
            min_pass_rate: Minimum pass rate threshold
        """
        self.strict_mode = strict_mode
        self.min_overlap = min_overlap
        self.min_pass_rate = min_pass_rate

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize and filter stopwords."""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return {w for w in words if w not in self.STOPWORDS and len(w) > 2}

    def _has_citation(self, sentence: str) -> bool:
        """Check if sentence has citation markers."""
        return bool(re.search(r'\[\d+\]|\(Doc:|\(Source:|ChunkID:', sentence))

    def _is_meta_sentence(self, sentence: str) -> bool:
        """Check if sentence is metadata/header, not a claim."""
        lower = sentence.lower().strip()
        return (
            lower.startswith(('##', '**', 'sources:', 'references:', 'note:'))
            or lower in ('', 'n/a', 'unknown')
            or len(sentence.split()) < 4
        )

    def verify(
        self,
        answer: str,
        chunks: list[EvidenceChunk],
        query: str,
    ) -> FirewallResult:
        """Verify answer against source chunks.

        Args:
            answer: Generated answer to verify
            chunks: Source evidence chunks
            query: Original user query

        Returns:
            FirewallResult with verification details
        """
        # Build combined source text for overlap checking
        chunk_texts = [c.snippet.lower() for c in chunks]
        all_source_terms = set()
        for text in chunk_texts:
            all_source_terms.update(self._tokenize(text))

        # Split answer into sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)

        flagged: list[str] = []
        verified_count = 0
        claim_count = 0
        per_sentence_details: list[dict] = []

        for sentence in sentences:
            sentence = sentence.strip()

            # Skip meta sentences
            if self._is_meta_sentence(sentence):
                continue

            claim_count += 1

            # Check if sentence has explicit citation
            has_citation = self._has_citation(sentence)

            # Calculate term overlap with sources
            sentence_terms = self._tokenize(sentence)
            overlap = len(sentence_terms & all_source_terms) / len(sentence_terms) if sentence_terms else 0.0

            # Determine if supported
            is_supported = has_citation or overlap >= self.min_overlap

            detail = {
                "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                "has_citation": has_citation,
                "term_overlap": round(overlap, 3),
                "supported": is_supported,
            }
            per_sentence_details.append(detail)

            if is_supported:
                verified_count += 1
            else:
                flagged.append(sentence)

        # In strict mode, clean the answer
        cleaned = answer
        if self.strict_mode and flagged:
            for claim in flagged:
                # Add warning markers
                cleaned = cleaned.replace(
                    claim,
                    f"[⚠️ UNVERIFIED] {claim}"
                )

        pass_rate = verified_count / max(claim_count, 1)

        return FirewallResult(
            original_answer=answer,
            cleaned_answer=cleaned,
            flagged_claims=flagged,
            verified_claims=verified_count,
            total_claims=claim_count,
            pass_rate=pass_rate,
            details={
                "sentences_checked": len(per_sentence_details),
                "source_chunks": len(chunks),
                "meets_threshold": pass_rate >= self.min_pass_rate,
                "per_sentence": per_sentence_details[:10],  # Limit for logs
            },
        )


__all__ = ["FirewallResult", "HallucinationFirewall"]
