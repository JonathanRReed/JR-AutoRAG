"""Heuristic evidence screening for generation.

Requires model to extract/quote spans before composing answers.
Rejects claims without matching evidence; lexical screening is not semantic entailment.

This module implements the "evidence-first contract" from SOTA RAG research:
- Answers are constrained to claims supported by retrieved passages
- Unsupported claims trigger additional retrieval or clarification
"""

from __future__ import annotations

import re
from .citation_verifier import CitationVerifier
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk


@dataclass
class ClaimVerification:
    """Result of verifying a single claim against evidence."""
    claim: str
    supported: bool
    supporting_span: str | None = None
    chunk_id: str | None = None
    confidence: float = 0.0


@dataclass
class EvidenceContractResult:
    """Result of evidence contract verification."""
    verified_claims: list[ClaimVerification]
    unsupported_claims: list[str]
    coverage_ratio: float
    pass_threshold: bool
    suggested_retrievals: list[str] = field(default_factory=list)


class EvidenceContract:
    """Enforce evidence-first generation contract.

    Verifies that generated claims have supporting evidence,
    and suggests additional retrieval for unsupported claims.

    Key features:
    - Extract factual claims from generated text
    - Find supporting spans in source chunks
    - Flag unsupported claims for verification
    - Generate targeted retrieval queries for gaps
    """

    # Patterns for identifying factual claims
    CLAIM_PATTERNS = [
        r'([A-Z][^.!?]*(?:is|are|was|were|has|have|will|can|should)[^.!?]+[.!?])',
        r'([A-Z][^.!?]*\d+[^.!?]+[.!?])',  # Claims with numbers
        r'([A-Z][^.!?]*(?:always|never|typically|often|usually)[^.!?]+[.!?])',
    ]

    # Stopwords to filter from overlap calculation
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

    def __init__(self, min_coverage: float = 0.7, min_overlap: float = 0.3):
        """Initialize evidence contract.

        Args:
            min_coverage: Minimum ratio of supported claims to pass (0-1)
            min_overlap: Minimum term overlap to consider a match (0-1)
        """
        self.min_coverage = min_coverage
        self.min_overlap = min_overlap
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.CLAIM_PATTERNS]

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize and filter stopwords."""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return {w for w in words if w not in self.STOPWORDS and len(w) > 2}

    def extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from generated text.

        Args:
            text: Generated answer text

        Returns:
            List of extracted claim sentences
        """
        claims = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            # Skip short sentences
            if len(sentence.split()) < 3:
                continue
            # Skip citations/references sections
            if sentence.lower().startswith(('sources:', 'references:', '##')):
                continue
            # Check for claim patterns
            for pattern in self._patterns:
                if pattern.search(sentence):
                    claims.append(sentence)
                    break

        return list(dict.fromkeys(claims))

    def find_supporting_span(
        self,
        claim: str,
        chunks: list[EvidenceChunk],
    ) -> tuple[str | None, str | None, float]:
        """Find a supporting span for a claim in the evidence chunks.

        Args:
            claim: The claim to find support for
            chunks: List of evidence chunks to search

        Returns:
            Tuple of (supporting_span, chunk_id, confidence)
        """
        clean_claim = claim
        for _, marker in CitationVerifier().extract_citations(claim):
            clean_claim = clean_claim.replace(marker, "")
        claim_terms = self._tokenize(clean_claim)
        if not claim_terms:
            return (None, None, 0.0)

        def numbers(text):
            return set(re.findall(r"(?<!\w)[+-]?\d+(?:[,.]\d+)*(?:%)?", text))
        negation = r"\b(?:not|no|never|without|cannot|can't|won't|isn't|aren't|doesn't|didn't)\b"
        claim_numbers = numbers(clean_claim)
        claim_negated = bool(re.search(negation, clean_claim, re.I))
        best_match = (None, None, 0.0)

        for chunk in chunks:
            snippet = chunk.snippet
            sentences = re.split(r'(?<=[.!?])\s+', snippet)

            for sentence in sentences:
                sent_terms = self._tokenize(sentence)
                if not sent_terms:
                    continue

                # Lexical overlap alone cannot establish entailment. Reject
                # obvious numerical and polarity mismatches conservatively.
                if not claim_numbers.issubset(numbers(sentence)):
                    continue
                if claim_negated != bool(re.search(negation, sentence, re.I)):
                    continue
                # This score is a lexical heuristic, not a probability of truth.
                overlap = len(claim_terms & sent_terms) / len(claim_terms | sent_terms)

                if overlap > best_match[2] and overlap > self.min_overlap:
                    best_match = (sentence.strip(), chunk.id, overlap)

        return best_match

    def verify_answer(
        self,
        answer: str,
        chunks: list[EvidenceChunk],
    ) -> EvidenceContractResult:
        """Verify that answer claims are supported by evidence.

        Args:
            answer: The generated answer to verify
            chunks: Source evidence chunks

        Returns:
            EvidenceContractResult with verification details
        """
        claims = self.extract_claims(answer)
        verified: list[ClaimVerification] = []
        unsupported: list[str] = []

        for claim in claims:
            verifier = CitationVerifier()
            citations = verifier.extract_citations(claim)
            valid_ids = verifier.get_valid_chunk_ids(chunks)
            valid_citations = all(value in valid_ids for value, _ in citations)
            cited_ids = {valid_ids[value] for value, _ in citations if value in valid_ids}
            candidate_chunks = ([chunk for chunk in chunks if chunk.id in cited_ids]
                                if citations else chunks)
            span, chunk_id, confidence = self.find_supporting_span(claim, candidate_chunks)
            is_supported = valid_citations and span is not None and confidence > 0.35

            verification = ClaimVerification(
                claim=claim,
                supported=is_supported,
                supporting_span=span,
                chunk_id=chunk_id,
                confidence=confidence if valid_citations else 0.0,
            )
            verified.append(verification)

            if not verification.supported:
                unsupported.append(claim)

        total_claims = len(claims)
        supported_claims = total_claims - len(unsupported)
        coverage = supported_claims / max(total_claims, 1)

        # Generate suggested retrievals for unsupported claims
        suggestions = self._generate_retrieval_suggestions(unsupported)

        return EvidenceContractResult(
            verified_claims=verified,
            unsupported_claims=unsupported,
            coverage_ratio=coverage,
            pass_threshold=coverage >= self.min_coverage,
            suggested_retrievals=suggestions,
        )

    def _generate_retrieval_suggestions(
        self,
        unsupported_claims: list[str],
    ) -> list[str]:
        """Generate targeted retrieval queries for unsupported claims."""
        suggestions = []

        for claim in unsupported_claims[:3]:  # Limit to 3 suggestions
            # Extract key terms (nouns, proper nouns, numbers)
            key_terms = []

            # Get capitalized words (likely proper nouns)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', claim)
            key_terms.extend(proper_nouns[:2])

            # Get longer words (likely meaningful terms)
            words = [w for w in claim.split() if len(w) > 5 and w.lower() not in self.STOPWORDS]
            key_terms.extend(words[:3])

            # Get numbers/dates
            numbers = re.findall(r'\b\d+[\d,\.]*\b', claim)
            key_terms.extend(numbers[:1])

            if key_terms:
                suggestions.append(" ".join(key_terms))

        return suggestions


__all__ = [
    "ClaimVerification",
    "EvidenceContractResult",
    "EvidenceContract",
]
