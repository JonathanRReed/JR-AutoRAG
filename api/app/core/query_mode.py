"""Query mode switch for grounded vs open-domain responses.

This module implements P0.1: Explicit query mode switch.
- GROUNDED: Answers only from corpus documents, never from LLM knowledge
- OPEN_DOMAIN: LLM can use general knowledge when corpus is insufficient

When grounded mode has no evidence, returns structured "no evidence" response
with suggested next actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryMode(str, Enum):
    """Query answering mode."""

    GROUNDED = "grounded"
    """Answer only from corpus documents. If no evidence found,
    return a structured 'no evidence' response instead of hallucinating."""

    OPEN_DOMAIN = "open_domain"
    """Allow LLM to use general knowledge when corpus is insufficient.
    Still prioritizes corpus evidence but can supplement with world knowledge."""


@dataclass
class SuggestedAction:
    """A suggested action when no evidence is found."""

    label: str
    description: str
    action_type: str  # "search_modification", "corpus_action", "mode_switch"

    def to_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "description": self.description,
            "action_type": self.action_type,
        }


@dataclass
class NoEvidenceResponse:
    """Structured response when grounded mode finds no supporting evidence.

    Provides clear communication that no corpus evidence was found,
    along with actionable suggestions for the user.
    """

    query: str
    message: str = "No supporting documents found for your query."
    suggested_actions: list[SuggestedAction] = field(default_factory=list)
    corpus_stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.suggested_actions:
            self.suggested_actions = self._default_suggestions()

    def _default_suggestions(self) -> list[SuggestedAction]:
        """Generate default suggestions based on query."""
        return [
            SuggestedAction(
                label="Try different keywords",
                description="Rephrase your question using different terms or synonyms",
                action_type="search_modification",
            ),
            SuggestedAction(
                label="Broaden your search",
                description="Remove specific constraints or use more general terms",
                action_type="search_modification",
            ),
            SuggestedAction(
                label="Upload relevant documents",
                description="Add documents that might contain the answer to your corpus",
                action_type="corpus_action",
            ),
            SuggestedAction(
                label="Switch to Open Domain mode",
                description="Allow the LLM to use general knowledge (may be less accurate)",
                action_type="mode_switch",
            ),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "found_evidence": False,
            "query": self.query,
            "message": self.message,
            "suggested_actions": [a.to_dict() for a in self.suggested_actions],
            "corpus_stats": self.corpus_stats,
        }


def build_no_evidence_answer(
    query: str,
    corpus_doc_count: int = 0,
    corpus_chunk_count: int = 0,
    search_terms_tried: list[str] | None = None,
) -> dict[str, Any]:
    """Build a complete answer response for no-evidence scenarios.

    This is used by the orchestrator when grounded mode finds no
    supporting documents.

    Args:
        query: Original user query
        corpus_doc_count: Number of documents in corpus
        corpus_chunk_count: Number of chunks in corpus
        search_terms_tried: List of search variations attempted

    Returns:
        Complete response dict compatible with QueryResponse schema
    """
    corpus_stats = {
        "doc_count": corpus_doc_count,
        "chunk_count": corpus_chunk_count,
    }
    if search_terms_tried:
        corpus_stats["search_terms_tried"] = search_terms_tried

    no_evidence = NoEvidenceResponse(
        query=query,
        corpus_stats=corpus_stats,
    )

    # Customize message based on corpus state
    if corpus_doc_count == 0:
        no_evidence.message = "Your corpus is empty. Please upload documents before querying."
        no_evidence.suggested_actions = [
            SuggestedAction(
                label="Upload documents",
                description="Add documents to your corpus to enable search",
                action_type="corpus_action",
            ),
            SuggestedAction(
                label="Load demo corpus",
                description="Try the demo dataset to explore features",
                action_type="corpus_action",
            ),
        ]
    elif corpus_chunk_count < 10:
        no_evidence.message = (
            f"Limited corpus ({corpus_doc_count} docs, {corpus_chunk_count} chunks). "
            "Consider adding more documents for better coverage."
        )

    return {
        "answer": no_evidence.message,
        "chunks": [],
        "sources": [],
        "grounding": {
            "grounded": False,
            "docs_used": 0,
            "citations_kept": 0,
            "chunks_dropped": 0,
            "mode": QueryMode.GROUNDED.value,
            "no_evidence_response": no_evidence.to_dict(),
        },
        "metrics": {
            "grounded": False,
            "mode": QueryMode.GROUNDED.value,
        },
    }


__all__ = [
    "QueryMode",
    "SuggestedAction",
    "NoEvidenceResponse",
    "build_no_evidence_answer",
]
