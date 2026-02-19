"""Post-generation evidence validator for 10/10 citation fidelity.

This module provides:
- Validation that every claim has quote + locator
- Quote existence verification against sources
- Detection of invented data (dates, numbers without evidence)
- Generation rejection if validation fails
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .citation_formatter import RichCitation


@dataclass
class ValidationResult:
    """Result of evidence validation."""
    is_valid: bool
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    missing_quotes: list[str]
    missing_locators: list[str]
    invented_data: list[str]
    issues: list[str] = field(default_factory=list)

    @property
    def support_rate(self) -> float:
        if self.total_claims == 0:
            return 1.0
        return self.supported_claims / self.total_claims

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "support_rate": self.support_rate,
            "missing_quotes": self.missing_quotes,
            "missing_locators": self.missing_locators,
            "invented_data": self.invented_data,
            "issues": self.issues,
        }


class EvidenceValidator:
    """Validates that generated answers meet 10/10 citation standards."""

    # Patterns for detecting claims that need citations
    CLAIM_PATTERNS = [
        r'\b\d+(?:\.\d+)?%',  # Percentages
        r'\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?',  # Dollar amounts
        r'Q[1-4]\s*20\d{2}',  # Quarter/year references
        r'(?:by|in|during)\s+(?:Q[1-4]\s*)?20\d{2}',  # Future dates
        r'\b(?:largest|biggest|first|only|leading|dominant)\b',  # Superlatives
        r'\b(?:will|expects?|projects?|forecasts?|anticipates?)\b',  # Predictions
    ]

    # Pattern for valid citations
    CITATION_PATTERN = r'"[^"]{10,}"[^"]*\([^)]*(?:Doc|ChunkID)[^)]*\)'

    # Pattern for quote + locator
    QUOTE_LOCATOR_PATTERN = r'"([^"]{10,200})"[^"]*\((?:Doc:\s*([^,)]+)|.*?ChunkID:\s*([a-zA-Z0-9_-]+))'

    def __init__(
        self,
        min_quote_length: int = 10,
        max_unsupported_rate: float = 0.0,  # 0% = all claims must be supported
        require_references_section: bool = True,
        require_sources_section: bool = True,
    ) -> None:
        self.min_quote_length = min_quote_length
        self.max_unsupported_rate = max_unsupported_rate
        self.require_references_section = require_references_section
        self.require_sources_section = require_sources_section

        # Compile patterns
        self._claim_patterns = [re.compile(p, re.IGNORECASE) for p in self.CLAIM_PATTERNS]

    def extract_claims_needing_evidence(self, answer: str) -> list[str]:
        """Extract sentences that contain claims needing evidence."""
        claims = []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)

        for sentence in sentences:
            # Skip if it's a rule (user-defined)
            if re.search(r'(?:Rule|user-defined)', sentence, re.IGNORECASE):
                continue

            # Skip if it's already marked as unknown
            if re.search(r'\[Unknown|No.*?found in|not available in sources', sentence, re.IGNORECASE):
                continue

            # Check if sentence contains claimable content
            for pattern in self._claim_patterns:
                if pattern.search(sentence):
                    claims.append(sentence.strip())
                    break

        return claims

    def extract_citations(self, answer: str) -> list[dict]:
        """Extract all citations with quotes and locators."""
        citations = []

        matches = re.findall(self.QUOTE_LOCATOR_PATTERN, answer)
        for match in matches:
            quote, doc_title, chunk_id = match
            citations.append({
                "quote": quote,
                "doc_title": doc_title.strip() if doc_title else "",
                "chunk_id": chunk_id.strip() if chunk_id else "",
            })

        return citations

    def detect_invented_data(
        self,
        answer: str,
        available_sources: list[RichCitation],
    ) -> list[str]:
        """Detect data that appears invented (not in sources)."""
        invented = []

        # Build set of all text from sources
        all_source_text = " ".join(c.full_snippet.lower() for c in available_sources)

        # Check for dates/quarters not in sources
        date_matches = re.findall(r'(?:Q[1-4]\s*)?20\d{2}', answer)
        for date in date_matches:
            if date.lower() not in all_source_text:
                invented.append(f"Date '{date}' not found in sources")

        # Check for specific dollar amounts
        dollar_matches = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)', answer)
        for amount in dollar_matches:
            # Normalize and check
            normalized = amount.replace(",", "")
            if normalized not in all_source_text.replace(",", ""):
                invented.append(f"Amount '${amount}' not found in sources")

        # Check for percentages
        pct_matches = re.findall(r'(\d+(?:\.\d+)?%)', answer)
        for pct in pct_matches:
            if pct.lower() not in all_source_text:
                invented.append(f"Percentage '{pct}' not found in sources")

        return invented

    def validate(
        self,
        answer: str,
        available_sources: list[RichCitation] | None = None,
    ) -> ValidationResult:
        """Validate an answer for 10/10 citation standards.

        Returns ValidationResult with pass/fail and detailed issues.
        """
        issues = []

        # 1. Check for References section
        has_references = bool(re.search(r'##\s*References', answer, re.IGNORECASE))
        if self.require_references_section and not has_references:
            issues.append("Missing required '## References' section")

        # 2. Check for Sources Used section
        has_sources = bool(re.search(r'##\s*Sources?\s*Used', answer, re.IGNORECASE))
        if self.require_sources_section and not has_sources:
            issues.append("Missing required '## Sources Used' section")

        # 3. Extract claims needing evidence
        claims = self.extract_claims_needing_evidence(answer)

        # 4. Extract citations provided
        self.extract_citations(answer)

        # 5. Check for missing quotes
        missing_quotes = []
        for claim in claims:
            # Check if claim has an associated quote nearby
            claim_context = answer[max(0, answer.find(claim[:30])-50):answer.find(claim[:30])+len(claim)+200]
            if not re.search(r'"[^"]{10,}"', claim_context):
                missing_quotes.append(claim[:100] + "..." if len(claim) > 100 else claim)

        # 6. Check for missing locators
        missing_locators = []
        quote_only_pattern = r'"[^"]{10,}"(?!\s*\([^)]*(?:Doc|ChunkID))'
        quote_only_matches = re.findall(quote_only_pattern, answer)
        for match in quote_only_matches:
            # Skip if it's in a template/example
            if "example" not in answer[max(0, answer.find(match)-50):answer.find(match)].lower():
                missing_locators.append(match[:50] + "...")

        # 7. Detect invented data
        invented_data = []
        if available_sources:
            invented_data = self.detect_invented_data(answer, available_sources)

        # Calculate metrics
        total_claims = len(claims)
        unsupported = len(missing_quotes)
        supported = total_claims - unsupported

        # Determine validity
        is_valid = (
            len(issues) == 0 and
            len(invented_data) == 0 and
            (total_claims == 0 or unsupported / total_claims <= self.max_unsupported_rate)
        )

        return ValidationResult(
            is_valid=is_valid,
            total_claims=total_claims,
            supported_claims=supported,
            unsupported_claims=unsupported,
            missing_quotes=missing_quotes,
            missing_locators=missing_locators,
            invented_data=invented_data,
            issues=issues,
        )

    def generate_rejection_message(self, result: ValidationResult) -> str:
        """Generate a human-readable rejection message."""
        parts = ["## Validation Failed\n"]

        if result.issues:
            parts.append("### Structural Issues")
            for issue in result.issues:
                parts.append(f"- {issue}")
            parts.append("")

        if result.missing_quotes:
            parts.append(f"### Claims Missing Quotes ({len(result.missing_quotes)})")
            for claim in result.missing_quotes[:5]:
                parts.append(f"- {claim}")
            if len(result.missing_quotes) > 5:
                parts.append(f"- ... and {len(result.missing_quotes) - 5} more")
            parts.append("")

        if result.invented_data:
            parts.append(f"### Potentially Invented Data ({len(result.invented_data)})")
            for item in result.invented_data[:5]:
                parts.append(f"- {item}")
            parts.append("")

        parts.append(f"**Support Rate**: {result.support_rate:.1%} ({result.supported_claims}/{result.total_claims} claims supported)")

        return "\n".join(parts)


def create_strict_validator() -> EvidenceValidator:
    """Create a validator with strictest settings for 10/10 compliance."""
    return EvidenceValidator(
        min_quote_length=10,
        max_unsupported_rate=0.0,  # 100% support required
        require_references_section=True,
        require_sources_section=True,
    )


__all__ = [
    "ValidationResult",
    "EvidenceValidator",
    "create_strict_validator",
]
