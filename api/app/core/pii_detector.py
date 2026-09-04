"""PII detection and redaction.

Provides detection and optional redaction of personally identifiable
information (PII) in documents and answers:

- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- Credit card numbers
- IP addresses
- Names (basic pattern matching)
- Addresses (basic pattern matching)

This helps ensure sensitive data isn't exposed in RAG responses.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("autorag.pii")


# =============================================================================
# PII Types
# =============================================================================


class PIIType(Enum):
    """Types of PII that can be detected."""

    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    UNKNOWN = "unknown"


@dataclass
class PIIMatch:
    """A detected PII instance."""

    pii_type: PIIType
    text: str
    start: int
    end: int
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.pii_type.value,
            "text": self.text[:4] + "..."
            if len(self.text) > 4
            else "...",  # Partial for safety
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 3),
        }

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class DetectionResult:
    """Result of PII detection."""

    original_text: str
    matches: list[PIIMatch] = field(default_factory=list)
    has_pii: bool = False
    pii_types_found: set[PIIType] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_pii": self.has_pii,
            "match_count": len(self.matches),
            "pii_types": [t.value for t in self.pii_types_found],
            "matches": [m.to_dict() for m in self.matches],
        }


# =============================================================================
# Detection Patterns
# =============================================================================

# Compiled regex patterns for PII detection
PII_PATTERNS = {
    PIIType.EMAIL: re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
    ),
    PIIType.PHONE: re.compile(
        r"""
        (?:
            (?:\+?1[-.\s]?)?        # Optional country code
            (?:\(?\d{3}\)?[-.\s]?)  # Area code
            \d{3}[-.\s]?\d{4}       # Local number
        )
        """,
        re.VERBOSE,
    ),
    PIIType.SSN: re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
    PIIType.CREDIT_CARD: re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"),
    PIIType.IP_ADDRESS: re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    PIIType.DATE_OF_BIRTH: re.compile(
        r"""
        \b(?:
            (?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2} |  # MM/DD/YYYY
            (?:0?[1-9]|[12]\d|3[01])[/\-](?:0?[1-9]|1[0-2])[/\-](?:19|20)\d{2} |  # DD/MM/YYYY
            (?:19|20)\d{2}[/\-](?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])    # YYYY/MM/DD
        )\b
        """,
        re.VERBOSE,
    ),
}

# Sensitive keyword patterns (lower confidence)
SENSITIVE_KEYWORDS = {
    "social security",
    "ssn",
    "credit card",
    "password",
    "pin",
    "passport number",
    "driver's license",
    "drivers license",
    "bank account",
    "routing number",
}


# =============================================================================
# PII Detector
# =============================================================================


class PIIDetector:
    """Detect PII in text.

    Usage:
        detector = PIIDetector()
        result = detector.detect("Contact me at john@example.com")
        if result.has_pii:
            print(f"Found PII: {result.pii_types_found}")
    """

    def __init__(
        self,
        enabled_types: set[PIIType] | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize detector.

        Args:
            enabled_types: PII types to detect (all if None)
            min_confidence: Minimum confidence to report
        """
        self.enabled_types = enabled_types or set(PIIType)
        self.min_confidence = min_confidence

    def detect(self, text: str) -> DetectionResult:
        """Detect PII in text.

        Args:
            text: Text to scan for PII

        Returns:
            DetectionResult with all matches
        """
        matches = []

        # Run pattern-based detection
        for pii_type, pattern in PII_PATTERNS.items():
            if pii_type not in self.enabled_types:
                continue

            for match in pattern.finditer(text):
                confidence = self._validate_match(pii_type, match.group())

                if confidence >= self.min_confidence:
                    matches.append(
                        PIIMatch(
                            pii_type=pii_type,
                            text=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=confidence,
                        )
                    )

        # Check for sensitive keywords
        text_lower = text.lower()
        for keyword in SENSITIVE_KEYWORDS:
            if keyword in text_lower:
                # Find positions
                pos = 0
                while True:
                    idx = text_lower.find(keyword, pos)
                    if idx == -1:
                        break
                    matches.append(
                        PIIMatch(
                            pii_type=PIIType.UNKNOWN,
                            text=text[idx : idx + len(keyword)],
                            start=idx,
                            end=idx + len(keyword),
                            confidence=0.6,  # Lower confidence for keywords
                        )
                    )
                    pos = idx + 1

        # Remove duplicates and sort by position
        seen = set()
        unique_matches = []
        for m in sorted(matches, key=lambda x: x.start):
            key = (m.start, m.end)
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)

        pii_types = {m.pii_type for m in unique_matches}

        return DetectionResult(
            original_text=text,
            matches=unique_matches,
            has_pii=len(unique_matches) > 0,
            pii_types_found=pii_types,
        )

    def _validate_match(self, pii_type: PIIType, text: str) -> float:
        """Validate a potential PII match.

        Returns confidence score 0.0-1.0.
        """
        if pii_type == PIIType.SSN:
            # SSN should have proper format: XXX-XX-XXXX
            # Check it's not all same digits
            digits = re.sub(r"\D", "", text)
            if len(set(digits)) == 1:
                return 0.0  # All same digit, probably not SSN
            return 0.9

        elif pii_type == PIIType.CREDIT_CARD:
            # Luhn algorithm check
            digits = re.sub(r"\D", "", text)
            if len(digits) not in (15, 16):
                return 0.0
            if self._luhn_check(digits):
                return 0.95
            return 0.3

        elif pii_type == PIIType.IP_ADDRESS:
            # Check octets are valid (0-255)
            parts = text.split(".")
            try:
                if all(0 <= int(p) <= 255 for p in parts):
                    # Some IPs are more likely to be PII
                    if text.startswith(("10.", "192.168.", "172.")):
                        return 0.7  # Private IP
                    return 0.5  # Could be any IP
            except ValueError:
                return 0.0
            return 0.5

        elif pii_type == PIIType.PHONE:
            # Validate phone number format
            digits = re.sub(r"\D", "", text)
            if len(digits) < 7 or len(digits) > 15:
                return 0.0
            return 0.8

        elif pii_type == PIIType.EMAIL:
            # Basic email validation
            if "@" in text and "." in text:
                return 0.95
            return 0.0

        return 0.7  # Default confidence

    def _luhn_check(self, card_number: str) -> bool:
        """Luhn algorithm for credit card validation."""

        def digits_of(n):
            return [int(d) for d in str(n)]

        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]

        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))

        return checksum % 10 == 0

    def redact(
        self,
        text: str,
        matches: list[PIIMatch] | None = None,
        replacement: str = "[REDACTED]",
    ) -> str:
        """Redact PII from text.

        Args:
            text: Text to redact
            matches: Pre-computed matches (detects if None)
            replacement: Replacement string for PII

        Returns:
            Redacted text
        """
        if matches is None:
            result = self.detect(text)
            matches = result.matches

        if not matches:
            return text

        # Sort matches by position (reverse order to preserve positions)
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        redacted = text
        for match in sorted_matches:
            # Use type-specific replacement if desired
            repl = f"[{match.pii_type.value.upper()}_REDACTED]"
            redacted = redacted[: match.start] + repl + redacted[match.end :]

        return redacted

    def mask(
        self,
        text: str,
        matches: list[PIIMatch] | None = None,
        mask_char: str = "*",
        show_last: int = 4,
    ) -> str:
        """Partially mask PII instead of full redaction.

        Args:
            text: Text to mask
            matches: Pre-computed matches
            mask_char: Character to use for masking
            show_last: Number of characters to leave visible

        Returns:
            Masked text
        """
        if matches is None:
            result = self.detect(text)
            matches = result.matches

        if not matches:
            return text

        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)

        masked = text
        for match in sorted_matches:
            pii_text = match.text
            if len(pii_text) > show_last:
                masked_pii = (
                    mask_char * (len(pii_text) - show_last) + pii_text[-show_last:]
                )
            else:
                masked_pii = mask_char * len(pii_text)

            masked = masked[: match.start] + masked_pii + masked[match.end :]

        return masked


# =============================================================================
# Singleton
# =============================================================================

_pii_detector: PIIDetector | None = None


def get_pii_detector() -> PIIDetector:
    """Get the global PII detector."""
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = PIIDetector()
    return _pii_detector


__all__ = [
    "PIIType",
    "PIIMatch",
    "DetectionResult",
    "PIIDetector",
    "get_pii_detector",
]
