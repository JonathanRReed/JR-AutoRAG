"""Claim verifier for post-generation answer verification.

This module provides:
- Claim extraction from generated answers
- Claim-to-span matching against retrieved evidence
- Verification results for uncited claims
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


@dataclass
class Claim:
    """A factual claim extracted from an answer."""
    text: str
    index: int  # Position in answer
    cited_refs: list[int] = field(default_factory=list)  # [1], [2] etc.


@dataclass
class SpanMatch:
    """A matching span in the retrieved evidence."""
    chunk_id: str
    start_char: int
    end_char: int
    matched_text: str
    similarity: float


@dataclass
class ClaimVerificationResult:
    """Result of verifying a single claim."""
    claim: Claim
    is_verified: bool
    supporting_span: SpanMatch | None
    confidence: float
    note: str = ""


@dataclass
class VerificationSummary:
    """Summary of full answer verification."""
    verified_claims: int
    unverified_claims: int
    total_claims: int
    coverage: float  # Fraction of claims verified
    results: list[ClaimVerificationResult]
    uncited_claims: list[str]


def extract_claims(answer: str) -> list[Claim]:
    """Extract factual claims from a generated answer.

    Claims are sentences or sentence fragments that make factual assertions.
    We also track any citation markers like [1], [2] that appear after them.
    """
    claims: list[Claim] = []

    # Split into sentences
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, answer.strip())

    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check for citations in the sentence
        citation_matches = re.findall(r'\[(\d+)\]', sentence)
        cited_refs = [int(c) for c in citation_matches]

        # Skip very short sentences or purely transitional phrases
        if len(sentence) < 20:
            continue

        # Skip sentences that are questions or purely connective
        if sentence.endswith('?'):
            continue

        # Skip common non-factual patterns
        skip_patterns = [
            r'^(However|Additionally|Moreover|Furthermore|In summary|In conclusion)[,:]?\s*$',
            r'^(Let me|I would|I can|I will)\b',
            r'^(As mentioned|As stated)\b',
        ]
        if any(re.match(p, sentence, re.I) for p in skip_patterns):
            continue

        claims.append(Claim(
            text=sentence,
            index=i,
            cited_refs=cited_refs,
        ))

    return claims


def find_best_span_match(
    claim_text: str,
    chunks: list[dict[str, Any]],
    min_similarity: float = 0.4,
) -> SpanMatch | None:
    """Find the best matching span for a claim in the retrieved chunks.

    Uses fuzzy matching to handle paraphrasing between source and answer.
    """
    # Clean claim text (remove citations)
    clean_claim = re.sub(r'\[\d+\]', '', claim_text).strip().lower()

    best_match: SpanMatch | None = None
    best_score = 0.0

    for chunk in chunks:
        chunk_text = chunk.get("text", chunk.get("chunk_text", "")).lower()
        chunk_id = chunk.get("chunk_id", chunk.get("id", ""))
        start_char = chunk.get("start_char", 0)
        end_char = chunk.get("end_char", 0)

        # Try to find overlapping content
        # 1. Check for direct substring match
        if clean_claim in chunk_text or chunk_text in clean_claim:
            best_match = SpanMatch(
                chunk_id=chunk_id,
                start_char=start_char,
                end_char=end_char,
                matched_text=chunk_text[:200],
                similarity=1.0,
            )
            break

        # 2. Use sequence matching for fuzzy comparison
        ratio = SequenceMatcher(None, clean_claim, chunk_text).ratio()

        if ratio > best_score and ratio >= min_similarity:
            best_score = ratio
            best_match = SpanMatch(
                chunk_id=chunk_id,
                start_char=start_char,
                end_char=end_char,
                matched_text=chunk_text[:200],
                similarity=ratio,
            )

        # 3. Check for key phrase overlap
        claim_words = set(clean_claim.split())
        chunk_words = set(chunk_text.split())
        if claim_words and chunk_words:
            overlap = len(claim_words & chunk_words) / len(claim_words)
            if overlap > best_score and overlap >= min_similarity:
                best_score = overlap
                best_match = SpanMatch(
                    chunk_id=chunk_id,
                    start_char=start_char,
                    end_char=end_char,
                    matched_text=chunk_text[:200],
                    similarity=overlap,
                )

    return best_match


def verify_claim(
    claim: Claim,
    chunks: list[dict[str, Any]],
    min_confidence: float = 0.4,
) -> ClaimVerificationResult:
    """Verify a single claim against retrieved evidence."""
    # Already has citations - assume verified if citation exists
    if claim.cited_refs:
        relevant_chunks = [
            c for c in chunks
            if any(ref == i + 1 for ref in claim.cited_refs for i, _ in enumerate(chunks))
        ]
        if relevant_chunks:
            chunk = relevant_chunks[0]
            return ClaimVerificationResult(
                claim=claim,
                is_verified=True,
                supporting_span=SpanMatch(
                    chunk_id=chunk.get("chunk_id", chunk.get("id", "")),
                    start_char=chunk.get("start_char", 0),
                    end_char=chunk.get("end_char", 0),
                    matched_text=chunk.get("text", "")[:200],
                    similarity=1.0,
                ),
                confidence=1.0,
                note="Has explicit citation",
            )

    # No citation - try to find supporting span
    span = find_best_span_match(claim.text, chunks, min_confidence)

    if span and span.similarity >= min_confidence:
        return ClaimVerificationResult(
            claim=claim,
            is_verified=True,
            supporting_span=span,
            confidence=span.similarity,
            note="Matched to span (uncited)",
        )

    return ClaimVerificationResult(
        claim=claim,
        is_verified=False,
        supporting_span=span,
        confidence=span.similarity if span else 0.0,
        note="No supporting evidence found",
    )


def verify_answer(
    answer: str,
    chunks: list[dict[str, Any]],
    min_confidence: float = 0.4,
) -> VerificationSummary:
    """Verify an entire answer against retrieved evidence.

    Returns a summary of claim verification including uncited claims
    that may need revision.
    """
    claims = extract_claims(answer)

    if not claims:
        return VerificationSummary(
            verified_claims=0,
            unverified_claims=0,
            total_claims=0,
            coverage=1.0,
            results=[],
            uncited_claims=[],
        )

    results: list[ClaimVerificationResult] = []
    uncited_claims: list[str] = []
    verified_count = 0

    for claim in claims:
        result = verify_claim(claim, chunks, min_confidence)
        results.append(result)

        if result.is_verified:
            verified_count += 1
        else:
            uncited_claims.append(claim.text)

    return VerificationSummary(
        verified_claims=verified_count,
        unverified_claims=len(claims) - verified_count,
        total_claims=len(claims),
        coverage=verified_count / len(claims) if claims else 1.0,
        results=results,
        uncited_claims=uncited_claims,
    )


def generate_revision_prompt(
    original_answer: str,
    uncited_claims: list[str],
) -> str:
    """Generate a revision prompt for uncited claims.

    This prompt can be appended to ask the model to either add citations
    or acknowledge uncertainty for uncited claims.
    """
    if not uncited_claims:
        return ""

    claims_list = "\n".join(f"- {claim}" for claim in uncited_claims[:5])

    return f"""
The following claims in your answer are not supported by the provided sources:

{claims_list}

Please revise your answer to either:
1. Add specific citations [1], [2], etc. for each claim if the information is in the sources
2. Clearly state uncertainty (e.g., "Based on the available information...") if the claim cannot be verified
3. Remove claims that cannot be substantiated from the provided sources

Revised answer:
"""


__all__ = [
    "Claim",
    "SpanMatch",
    "ClaimVerificationResult",
    "VerificationSummary",
    "extract_claims",
    "find_best_span_match",
    "verify_claim",
    "verify_answer",
    "generate_revision_prompt",
]
