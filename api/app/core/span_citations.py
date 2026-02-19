"""Span-level citation system for precise grounding.

This module extends the citation verifier with:
- Character-level span citations (start_char, end_char in source)
- Claim extraction and mapping to supporting spans
- Citation audit view for UI
- Coverage analysis for unsupported claims

Implements the requirement: "Citation for every non-trivial claim with
character-level offset into the retrieved chunk."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk


# =============================================================================
# Span Citation Types
# =============================================================================

@dataclass
class SpanCitation:
    """A citation that points to a specific span in a source chunk."""
    chunk_id: str
    start_char: int
    end_char: int
    quoted_text: str
    confidence: float = 1.0  # How confident we are in this match

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "quoted_text": self.quoted_text[:100] + "..." if len(self.quoted_text) > 100 else self.quoted_text,
            "confidence": round(self.confidence, 3),
        }

    @property
    def span_length(self) -> int:
        return self.end_char - self.start_char


@dataclass
class ExtractedClaim:
    """A claim extracted from the answer."""
    text: str
    start_pos: int  # Position in answer
    end_pos: int
    is_trivial: bool = False  # e.g., "The answer is..."

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "is_trivial": self.is_trivial,
        }


@dataclass
class ClaimMapping:
    """Mapping between a claim and its supporting evidence spans."""
    claim: ExtractedClaim
    supporting_spans: list[SpanCitation] = field(default_factory=list)
    unsupported: bool = False
    support_strength: float = 0.0  # 0-1, how well supported

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "supporting_spans": [s.to_dict() for s in self.supporting_spans],
            "unsupported": self.unsupported,
            "support_strength": round(self.support_strength, 3),
        }


# =============================================================================
# Validation Result
# =============================================================================

@dataclass
class SpanValidationResult:
    """Result of span-level citation validation."""
    answer: str
    claims: list[ExtractedClaim]
    claim_mappings: list[ClaimMapping]

    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    trivial_claims: int = 0

    overall_coverage: float = 0.0  # % of non-trivial claims supported

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "trivial_claims": self.trivial_claims,
            "overall_coverage": round(self.overall_coverage, 3),
            "claim_mappings": [m.to_dict() for m in self.claim_mappings],
        }

    @property
    def is_fully_supported(self) -> bool:
        return self.unsupported_claims == 0


# =============================================================================
# Claim Extractor
# =============================================================================

class ClaimExtractor:
    """Extract factual claims from an answer."""

    # Patterns for trivial claims that don't need citations
    TRIVIAL_PATTERNS = [
        r'^(Yes|No|The answer is|In summary|Therefore|Thus|So|Consequently)[,\s]',
        r'^(Based on|According to|As mentioned)[,\s]',
        r'^(I think|I believe|It seems)[,\s]',
        r'^\d+\.',  # List numbering
    ]

    def __init__(self) -> None:
        self._trivial_patterns = [re.compile(p) for p in self.TRIVIAL_PATTERNS]

    def extract_claims(self, answer: str) -> list[ExtractedClaim]:
        """Extract claims from an answer.

        Uses sentence splitting with consideration for:
        - Multiple sentences per claim
        - Trivial transitional phrases
        - List items
        """
        claims = []

        # Split into sentences
        sentences = self._split_sentences(answer)

        pos = 0
        for sentence in sentences:
            # Find position in original text
            start = answer.find(sentence, pos)
            if start == -1:
                start = pos
            end = start + len(sentence)
            pos = end

            # Check if trivial
            is_trivial = self._is_trivial(sentence)

            # Only add non-empty sentences
            if sentence.strip():
                claims.append(ExtractedClaim(
                    text=sentence.strip(),
                    start_pos=start,
                    end_pos=end,
                    is_trivial=is_trivial,
                ))

        return claims

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting on . ! ?
        # Handles abbreviations reasonably
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_trivial(self, sentence: str) -> bool:
        """Check if a sentence is trivial (doesn't need citation)."""
        sentence = sentence.strip()

        # Very short sentences are often trivial
        if len(sentence) < 20:
            return True

        # Check patterns
        return any(pattern.match(sentence) for pattern in self._trivial_patterns)


# =============================================================================
# Span Matcher
# =============================================================================

class SpanMatcher:
    """Match claims to source spans."""

    def __init__(
        self,
        min_match_ratio: float = 0.5,
        min_span_length: int = 20,
    ) -> None:
        self.min_match_ratio = min_match_ratio
        self.min_span_length = min_span_length

    def find_supporting_spans(
        self,
        claim: str,
        chunks: list[EvidenceChunk],
    ) -> list[SpanCitation]:
        """Find spans in chunks that support a claim.

        Uses fuzzy matching to find similar text in source chunks.
        """
        spans = []

        # Normalize claim for matching
        claim_normalized = self._normalize(claim)

        for chunk in chunks:
            chunk_id = getattr(chunk, 'id', None) or getattr(chunk, 'chunk_id', 'unknown')
            chunk_text = chunk.text

            # Try to find matching spans
            match_spans = self._find_matches(claim_normalized, chunk_text)

            for start, end, confidence in match_spans:
                if end - start >= self.min_span_length:
                    spans.append(SpanCitation(
                        chunk_id=chunk_id,
                        start_char=start,
                        end_char=end,
                        quoted_text=chunk_text[start:end],
                        confidence=confidence,
                    ))

        # Sort by confidence and return best matches
        spans.sort(key=lambda x: x.confidence, reverse=True)
        return spans[:3]  # Max 3 supporting spans per claim

    def _normalize(self, text: str) -> str:
        """Normalize text for matching."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def _find_matches(
        self,
        needle: str,
        haystack: str,
    ) -> list[tuple[int, int, float]]:
        """Find matching spans with confidence scores.

        Returns list of (start, end, confidence) tuples.
        """
        matches = []

        # Normalize haystack
        haystack_normalized = self._normalize(haystack)

        # Split needle into chunks for partial matching
        words = needle.split()

        if len(words) < 3:
            # Too short for reliable matching
            return []

        # Sliding window matching
        window_size = min(len(words), 10)

        for i in range(len(words) - window_size + 1):
            window = ' '.join(words[i:i + window_size])

            # Find in normalized text
            pos = haystack_normalized.find(window)
            if pos != -1:
                # Map position back to original text
                start = self._map_position(haystack, haystack_normalized, pos)
                end = self._map_position(haystack, haystack_normalized, pos + len(window))

                # Calculate confidence based on match length
                confidence = len(window) / len(needle)
                matches.append((start, end, confidence))

        # Also try sequence matching for fuzzy matches
        matcher = SequenceMatcher(None, needle, haystack_normalized)
        blocks = matcher.get_matching_blocks()

        for block in blocks:
            if block.size >= 20:  # Minimum match size
                start = self._map_position(haystack, haystack_normalized, block.b)
                end = self._map_position(haystack, haystack_normalized, block.b + block.size)
                confidence = block.size / len(needle)

                if confidence >= self.min_match_ratio:
                    matches.append((start, end, confidence))

        # Deduplicate and merge overlapping
        return self._merge_overlapping(matches)

    def _map_position(
        self,
        original: str,
        normalized: str,
        norm_pos: int,
    ) -> int:
        """Map position in normalized text back to original."""
        # Simple approximation - in practice would need character mapping
        ratio = len(original) / len(normalized) if normalized else 1
        return int(norm_pos * ratio)

    def _merge_overlapping(
        self,
        matches: list[tuple[int, int, float]],
    ) -> list[tuple[int, int, float]]:
        """Merge overlapping matches."""
        if not matches:
            return []

        # Sort by start position
        sorted_matches = sorted(matches, key=lambda x: x[0])

        merged = [sorted_matches[0]]
        for start, end, conf in sorted_matches[1:]:
            last_start, last_end, last_conf = merged[-1]

            if start <= last_end:
                # Overlapping - merge
                merged[-1] = (last_start, max(end, last_end), max(conf, last_conf))
            else:
                merged.append((start, end, conf))

        return merged


# =============================================================================
# Span Citation Validator
# =============================================================================

class SpanCitationValidator:
    """Validate citations at the span level."""

    def __init__(
        self,
        min_support_strength: float = 0.3,
    ) -> None:
        self.min_support_strength = min_support_strength
        self._extractor = ClaimExtractor()
        self._matcher = SpanMatcher()

    def validate(
        self,
        answer: str,
        chunks: list[EvidenceChunk],
    ) -> SpanValidationResult:
        """Validate that all claims are supported by source spans.

        Args:
            answer: Generated answer
            chunks: Source evidence chunks

        Returns:
            SpanValidationResult with claim-level analysis
        """
        # Extract claims
        claims = self._extractor.extract_claims(answer)

        # Map each claim to supporting spans
        mappings = []
        supported = 0
        unsupported = 0
        trivial = 0

        for claim in claims:
            if claim.is_trivial:
                trivial += 1
                mappings.append(ClaimMapping(
                    claim=claim,
                    support_strength=1.0,  # Trivial claims are "supported"
                ))
                continue

            # Find supporting spans
            spans = self._matcher.find_supporting_spans(claim.text, chunks)

            # Calculate support strength
            support_strength = max(s.confidence for s in spans) if spans else 0.0

            is_unsupported = support_strength < self.min_support_strength

            if is_unsupported:
                unsupported += 1
            else:
                supported += 1

            mappings.append(ClaimMapping(
                claim=claim,
                supporting_spans=spans,
                unsupported=is_unsupported,
                support_strength=support_strength,
            ))

        # Calculate overall coverage
        non_trivial = len(claims) - trivial
        coverage = supported / non_trivial if non_trivial > 0 else 1.0

        return SpanValidationResult(
            answer=answer,
            claims=claims,
            claim_mappings=mappings,
            total_claims=len(claims),
            supported_claims=supported,
            unsupported_claims=unsupported,
            trivial_claims=trivial,
            overall_coverage=coverage,
        )

    def get_audit_view(
        self,
        result: SpanValidationResult,
        format: str = "markdown",
    ) -> str:
        """Generate an audit view for the validation result.

        Args:
            result: Validation result
            format: Output format ("markdown" or "html")

        Returns:
            Formatted audit view
        """
        if format == "html":
            return self._generate_html_audit(result)
        return self._generate_markdown_audit(result)

    def _generate_markdown_audit(self, result: SpanValidationResult) -> str:
        """Generate markdown audit view."""
        lines = [
            "# Citation Audit",
            "",
            f"**Overall Coverage:** {result.overall_coverage:.0%}",
            f"**Supported Claims:** {result.supported_claims}/{result.total_claims - result.trivial_claims}",
            f"**Unsupported Claims:** {result.unsupported_claims}",
            "",
            "## Claim Analysis",
            "",
        ]

        for i, mapping in enumerate(result.claim_mappings, 1):
            claim = mapping.claim

            if claim.is_trivial:
                status = "✓ Trivial"
            elif mapping.unsupported:
                status = "❌ Unsupported"
            else:
                status = "✓ Supported"

            lines.append(f"### Claim {i}: {status}")
            lines.append(f"> {claim.text}")
            lines.append("")

            if mapping.supporting_spans and not claim.is_trivial:
                lines.append("**Supporting evidence:**")
                for span in mapping.supporting_spans:
                    lines.append(f"- [{span.chunk_id}:{span.start_char}-{span.end_char}] \"{span.quoted_text[:80]}...\"")
                lines.append("")

        return "\n".join(lines)

    def _generate_html_audit(self, result: SpanValidationResult) -> str:
        """Generate HTML audit view with highlighting."""
        html_parts = [
            "<div class='citation-audit'>",
            "<h2>Citation Audit</h2>",
            f"<p><strong>Coverage:</strong> {result.overall_coverage:.0%}</p>",
            "<div class='answer'>",
        ]

        # Reconstruct answer with highlighting
        answer = result.answer
        pos = 0

        for mapping in result.claim_mappings:
            claim = mapping.claim

            # Add text before claim
            if claim.start_pos > pos:
                html_parts.append(f"<span>{answer[pos:claim.start_pos]}</span>")

            # Add claim with appropriate class
            if claim.is_trivial:
                css_class = "trivial"
            elif mapping.unsupported:
                css_class = "unsupported"
            else:
                css_class = "supported"

            html_parts.append(
                f"<span class='claim {css_class}'>{answer[claim.start_pos:claim.end_pos]}</span>"
            )

            pos = claim.end_pos

        # Add remaining text
        if pos < len(answer):
            html_parts.append(f"<span>{answer[pos:]}</span>")

        html_parts.extend([
            "</div>",
            "</div>",
        ])

        return "\n".join(html_parts)


# =============================================================================
# Singleton
# =============================================================================

_span_validator: SpanCitationValidator | None = None


def get_span_validator() -> SpanCitationValidator:
    """Get the global span citation validator."""
    global _span_validator
    if _span_validator is None:
        _span_validator = SpanCitationValidator()
    return _span_validator


__all__ = [
    "SpanCitation",
    "ExtractedClaim",
    "ClaimMapping",
    "SpanValidationResult",
    "ClaimExtractor",
    "SpanMatcher",
    "SpanCitationValidator",
    "get_span_validator",
]
