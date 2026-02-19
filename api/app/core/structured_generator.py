"""Two-pass structured generator for 10/10 citation fidelity.

This module provides:
- Pass 1: Generate structured claim skeleton
- Pass 2: Fill slots only with evidence from KB
- No invented data - empty slots marked as "Unknown"
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .citation_formatter import RichCitation


class ClaimType(str, Enum):
    """Types of claims that need evidence."""
    MOAT = "moat"
    FAILURE = "failure_mode"
    CATALYST = "catalyst"
    RISK_RULE = "risk_rule"
    METRIC = "metric"
    FACT = "fact"


@dataclass
class ClaimSlot:
    """A slot for a claim that needs evidence."""
    claim_type: ClaimType
    claim_text: str = ""
    evidence_quote: str = ""
    evidence_locator: str = ""  # "Doc: Title, ChunkID: xxx"
    chunk_id: str = ""
    is_user_defined: bool = False  # For rules that don't need citations
    is_unknown: bool = False

    @property
    def is_filled(self) -> bool:
        """Check if slot has valid evidence."""
        if self.is_user_defined:
            return bool(self.claim_text)
        return bool(self.claim_text and self.evidence_quote and self.evidence_locator)

    def to_formatted_line(self) -> str:
        """Format as audit-ready output line."""
        if self.is_unknown:
            return f"**{self.claim_type.value.title()}**: [Unknown based on current knowledge base]"

        if self.is_user_defined:
            return f"**{self.claim_type.value.title()}** (user-defined): {self.claim_text}"

        if not self.is_filled:
            return f"**{self.claim_type.value.title()}**: [No evidence found in sources]"

        return f'**{self.claim_type.value.title()}**: {self.claim_text} "{self.evidence_quote}" ({self.evidence_locator})'


@dataclass
class StructuredPick:
    """A structured pick with claim slots."""
    identifier: str  # e.g., ticker symbol
    name: str  # e.g., company name
    theme: str = ""
    slots: list[ClaimSlot] = field(default_factory=list)

    def add_slot(self, claim_type: ClaimType, is_user_defined: bool = False) -> ClaimSlot:
        slot = ClaimSlot(claim_type=claim_type, is_user_defined=is_user_defined)
        self.slots.append(slot)
        return slot

    def get_slot(self, claim_type: ClaimType) -> ClaimSlot | None:
        for slot in self.slots:
            if slot.claim_type == claim_type:
                return slot
        return None

    def to_formatted_section(self) -> str:
        """Format as audit-ready output section."""
        lines = [f"### {self.name} ({self.identifier})"]
        if self.theme:
            lines.append(f"*Theme: {self.theme}*\n")

        for slot in self.slots:
            lines.append(slot.to_formatted_line())

        return "\n".join(lines)


@dataclass
class StructuredMemo:
    """A structured memo with multiple picks."""
    title: str = ""
    picks: list[StructuredPick] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)

    def add_pick(self, identifier: str, name: str, theme: str = "") -> StructuredPick:
        pick = StructuredPick(identifier=identifier, name=name, theme=theme)
        self.picks.append(pick)
        return pick

    def to_formatted_memo(self) -> str:
        """Generate the final formatted memo."""
        lines = []

        if self.title:
            lines.append(f"# {self.title}\n")

        for pick in self.picks:
            lines.append(pick.to_formatted_section())
            lines.append("")

        # Add references section
        lines.append("## References\n")
        ref_num = 1
        for pick in self.picks:
            for slot in pick.slots:
                if slot.is_filled and not slot.is_user_defined:
                    lines.append(f'[{ref_num}] {slot.evidence_locator} - "{slot.evidence_quote[:60]}..."')
                    ref_num += 1

        # Add sources used section
        lines.append("\n## Sources Used\n")
        if self.sources_used:
            lines.append(f"This memo used only: {', '.join(self.sources_used)}. No external knowledge was used.")
        else:
            lines.append("No sources were retrieved for this memo.")

        return "\n".join(lines)


class StructuredGenerator:
    """Two-pass generator for strict evidence grounding."""

    # Mapping of claim types to retrieval query templates
    QUERY_TEMPLATES = {
        ClaimType.MOAT: [
            "{identifier} competitive advantage moat",
            "{identifier} market position differentiation",
        ],
        ClaimType.FAILURE: [
            "{identifier} risks headwinds challenges",
            "{identifier} regulatory competition threats",
        ],
        ClaimType.CATALYST: [
            "{identifier} catalyst growth driver 2025 2026",
            "{identifier} upcoming events milestones",
        ],
        ClaimType.METRIC: [
            "{identifier} revenue profit margin metrics",
            "{identifier} financial performance numbers",
        ],
    }

    def __init__(
        self,
        retrieval_fn: Callable[[str], list[RichCitation]] | None = None,
        min_quote_length: int = 10,
        max_quote_length: int = 100,
    ) -> None:
        self.retrieval_fn = retrieval_fn
        self.min_quote_length = min_quote_length
        self.max_quote_length = max_quote_length

    def create_skeleton(
        self,
        identifiers: list[dict[str, str]],  # [{"id": "CEG", "name": "Constellation Energy", "theme": "Nuclear"}]
        claim_types: list[ClaimType] | None = None,
    ) -> StructuredMemo:
        """Pass 1: Create skeleton with empty claim slots."""
        if claim_types is None:
            claim_types = [ClaimType.MOAT, ClaimType.FAILURE, ClaimType.CATALYST, ClaimType.RISK_RULE]

        memo = StructuredMemo()

        for item in identifiers:
            pick = memo.add_pick(
                identifier=item.get("id", ""),
                name=item.get("name", ""),
                theme=item.get("theme", ""),
            )

            for claim_type in claim_types:
                is_user_defined = claim_type == ClaimType.RISK_RULE
                pick.add_slot(claim_type, is_user_defined=is_user_defined)

        return memo

    def extract_best_quote(
        self,
        text: str,
        claim_type: ClaimType,
    ) -> str:
        """Extract the most relevant quote for a claim type."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Keywords by claim type
        keywords = {
            ClaimType.MOAT: ["advantage", "moat", "leading", "dominant", "market share", "differentiat"],
            ClaimType.FAILURE: ["risk", "challenge", "headwind", "threat", "competition", "regulatory"],
            ClaimType.CATALYST: ["catalyst", "growth", "driver", "milestone", "announcement", "launch"],
            ClaimType.METRIC: ["revenue", "profit", "margin", "growth", "percent", "$"],
        }

        target_keywords = keywords.get(claim_type, [])

        # Score sentences by keyword relevance
        scored = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for kw in target_keywords if kw in sentence_lower)
            if score > 0 and len(sentence) >= self.min_quote_length:
                scored.append((score, sentence))

        # Return best match
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            # Truncate if too long
            if len(best) > self.max_quote_length:
                best = best[:self.max_quote_length] + "..."
            return best.strip()

        # Fallback: first sentence that's long enough
        for sentence in sentences:
            if len(sentence) >= self.min_quote_length:
                return sentence[:self.max_quote_length].strip()

        return ""

    def fill_slot(
        self,
        slot: ClaimSlot,
        citations: list[RichCitation],
    ) -> bool:
        """Pass 2: Fill a single slot with evidence if available."""
        if slot.is_user_defined:
            # User-defined slots don't need evidence
            return True

        if not citations:
            slot.is_unknown = True
            return False

        # Use the highest-scored citation
        best_citation = max(citations, key=lambda c: c.relevance_score)

        # Extract best quote for this claim type
        quote = self.extract_best_quote(best_citation.full_snippet, slot.claim_type)

        if not quote:
            slot.is_unknown = True
            return False

        # Fill the slot
        slot.evidence_quote = quote
        slot.evidence_locator = f"Doc: {best_citation.document_title}, ChunkID: {best_citation.chunk_id}"
        slot.chunk_id = best_citation.chunk_id

        # Generate claim text based on evidence
        slot.claim_text = self._generate_claim_from_quote(quote, slot.claim_type)

        return True

    def _generate_claim_from_quote(self, quote: str, claim_type: ClaimType) -> str:
        """Generate a claim sentence from a quote."""
        # Simple extraction - in production, this could use LLM
        prefixes = {
            ClaimType.MOAT: "Competitive advantage includes",
            ClaimType.FAILURE: "Key risk factor is",
            ClaimType.CATALYST: "Potential catalyst is",
            ClaimType.METRIC: "Key metric shows",
        }
        prefix = prefixes.get(claim_type, "Evidence shows")

        # Clean the quote
        clean_quote = quote.strip().rstrip(".")

        return f"{prefix} that {clean_quote.lower()[:50]}..." if len(clean_quote) > 50 else f"{prefix} that {clean_quote.lower()}."

    def fill_memo(
        self,
        memo: StructuredMemo,
        citations_by_identifier: dict[str, dict[ClaimType, list[RichCitation]]],
    ) -> StructuredMemo:
        """Pass 2: Fill all slots in the memo with evidence."""
        sources_used = set()

        for pick in memo.picks:
            identifier_citations = citations_by_identifier.get(pick.identifier, {})

            for slot in pick.slots:
                if slot.is_user_defined:
                    continue

                claim_citations = identifier_citations.get(slot.claim_type, [])
                self.fill_slot(slot, claim_citations)

                # Track sources
                for cit in claim_citations:
                    sources_used.add(cit.document_title)

        memo.sources_used = sorted(sources_used)
        return memo

    def generate_targeted_queries(
        self,
        pick: StructuredPick,
    ) -> dict[ClaimType, list[str]]:
        """Generate targeted sub-queries for each claim type."""
        queries = {}

        for slot in pick.slots:
            if slot.is_user_defined:
                continue

            templates = self.QUERY_TEMPLATES.get(slot.claim_type, [])
            queries[slot.claim_type] = [
                t.format(identifier=pick.identifier, name=pick.name)
                for t in templates
            ]

        return queries


def create_investment_memo_skeleton(
    picks: list[dict[str, str]],
) -> StructuredMemo:
    """Create a skeleton for an investment memo."""
    generator = StructuredGenerator()

    memo = generator.create_skeleton(
        identifiers=picks,
        claim_types=[
            ClaimType.MOAT,
            ClaimType.FAILURE,
            ClaimType.CATALYST,
            ClaimType.RISK_RULE,
        ],
    )
    memo.title = "Investment Memo"

    return memo


__all__ = [
    "ClaimType",
    "ClaimSlot",
    "StructuredPick",
    "StructuredMemo",
    "StructuredGenerator",
    "create_investment_memo_skeleton",
]
