"""Lightweight classifier for determining if a query needs RAG.

This module implements P0.4: Non-RAG Prompt Detection.
- Quickly classify whether a query can be answered from corpus
- Determine if graph/gatherer are needed
- Default: skip heavy operations unless necessary
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class QueryCategory(str, Enum):
    """Category of query for routing decisions."""

    CHITCHAT = "chitchat"           # Greetings, thanks, simple interaction
    CORPUS_REQUIRED = "corpus_required"  # Factual Q that needs documents
    CORPUS_OPTIONAL = "corpus_optional"  # May benefit from docs but not required
    META_QUERY = "meta_query"       # Query about the system itself
    COMPLEX_ANALYSIS = "complex_analysis"  # Needs graph/multi-hop reasoning


@dataclass
class ClassifyResult:
    """Result of corpus classification."""

    category: QueryCategory
    needs_corpus: bool
    needs_graph: bool
    needs_gatherer: bool
    confidence: float
    reasoning: str | None = None

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "needs_corpus": self.needs_corpus,
            "needs_graph": self.needs_graph,
            "needs_gatherer": self.needs_gatherer,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


class CorpusClassifier:
    """Classify whether a query needs RAG or can be answered directly.

    Uses rule-based patterns for fast classification, with optional
    LLM-based refinement for ambiguous cases.

    Default behavior:
    - No graph build unless query indicates need
    - No gatherer unless corpus lookup needed
    - Fast path for chitchat/meta queries
    """

    # Chitchat patterns - don't need corpus
    CHITCHAT_PATTERNS = [
        r"^(hi|hello|hey|greetings|good\s+(?:morning|afternoon|evening))[\s!.,]*(there|everyone|all)?[\s!.,]*$",
        r"^(thanks|thank\s+you|thx|ty)[\s!.,]*",
        r"^(bye|goodbye|see\s+you|later|have\s+a\s+nice\s+day)[\s!.,]*$",
        r"^how\s+are\s+you",
        r"^what['\"]?s\s+up",
        r"^(ok|okay|sure|got\s+it|understood)[\s!.,]*$",
        r"^(yes|no|yep|nope|yea|nah)[\s!.,]*$",
        r"^(hi|hello|hey)[\s,]+.{0,5}$",  # Short greetings with 1-2 extra words
    ]

    # Meta patterns - queries about the system
    META_PATTERNS = [
        r"what\s+(?:can\s+you|do\s+you)\s+do",
        r"how\s+(?:do\s+(?:you|i)|does\s+this)\s+work",
        r"what\s+(?:are\s+you|is\s+this)",
        r"(?:help|commands|features|capabilities)",
        r"what\s+documents?\s+(?:do\s+(?:you|i)\s+have|are\s+(?:available|loaded))",
        r"(?:how\s+many|list|show)\s+(?:documents?|files?|sources?)",
    ]

    # Patterns indicating need for corpus retrieval
    CORPUS_INDICATORS = [
        r"(?:what|who|when|where|which|how)\s+(?:is|are|was|were|did|does|do)",
        r"(?:tell|explain|describe|summarize)\s+(?:me|us)\s+(?:about|what)",
        r"(?:according|based)\s+(?:to|on)\s+(?:the|your|my)",
        r"(?:find|search|look\s+up|retrieve)",
        r"(?:in\s+the|from\s+the)\s+(?:document|file|text|corpus|source)",
        r"(?:quote|cite|reference|mention)",
    ]

    # Patterns indicating complex multi-hop reasoning (needs graph)
    COMPLEX_INDICATORS = [
        r"(?:and\s+also|as\s+well\s+as|in\s+addition)",
        r"(?:compare|contrast|difference\s+between)",
        r"(?:relationship|connection|link)\s+between",
        r"(?:how\s+does|why\s+does).*(?:relate|affect|impact|influence)",
        r"(?:across|between|among)\s+(?:multiple|different|various)",
        r"(?:timeline|chronology|sequence|history)",
        r"(?:all|every|each)\s+(?:of\s+the|mention)",
    ]

    def __init__(self) -> None:
        # Compile regex patterns for efficiency
        self._chitchat_re = [re.compile(p, re.IGNORECASE) for p in self.CHITCHAT_PATTERNS]
        self._meta_re = [re.compile(p, re.IGNORECASE) for p in self.META_PATTERNS]
        self._corpus_re = [re.compile(p, re.IGNORECASE) for p in self.CORPUS_INDICATORS]
        self._complex_re = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_INDICATORS]

    def _matches_any(self, text: str, patterns: list[re.Pattern]) -> tuple[bool, str | None]:
        """Check if text matches any pattern."""
        for pattern in patterns:
            if pattern.search(text):
                return True, pattern.pattern
        return False, None

    def classify(self, query: str) -> ClassifyResult:
        """Classify a query to determine RAG requirements.

        Args:
            query: User's question

        Returns:
            ClassifyResult with routing decisions
        """
        query = query.strip()

        # Empty or very short queries
        if len(query) < 3:
            return ClassifyResult(
                category=QueryCategory.CHITCHAT,
                needs_corpus=False,
                needs_graph=False,
                needs_gatherer=False,
                confidence=0.95,
                reasoning="Query too short",
            )

        # Check for chitchat
        is_chitchat, pattern = self._matches_any(query, self._chitchat_re)
        if is_chitchat:
            return ClassifyResult(
                category=QueryCategory.CHITCHAT,
                needs_corpus=False,
                needs_graph=False,
                needs_gatherer=False,
                confidence=0.9,
                reasoning="Matches chitchat pattern",
            )

        # Check for meta queries
        is_meta, pattern = self._matches_any(query, self._meta_re)
        if is_meta:
            return ClassifyResult(
                category=QueryCategory.META_QUERY,
                needs_corpus=False,  # May need corpus stats but not retrieval
                needs_graph=False,
                needs_gatherer=False,
                confidence=0.85,
                reasoning="Matches meta query pattern",
            )

        # Check for complex analysis (needs graph)
        is_complex, pattern = self._matches_any(query, self._complex_re)
        needs_corpus, _ = self._matches_any(query, self._corpus_re)

        if is_complex:
            return ClassifyResult(
                category=QueryCategory.COMPLEX_ANALYSIS,
                needs_corpus=True,
                needs_graph=True,  # Complex queries benefit from graph
                needs_gatherer=True,
                confidence=0.8,
                reasoning="Complex multi-hop query detected",
            )

        # Check for corpus-requiring queries
        if needs_corpus:
            return ClassifyResult(
                category=QueryCategory.CORPUS_REQUIRED,
                needs_corpus=True,
                needs_graph=False,  # Simple retrieval, no graph needed
                needs_gatherer=True,
                confidence=0.85,
                reasoning="Query indicates need for document retrieval",
            )

        # Default: query likely benefits from corpus but could work without
        # Use standard retrieval but skip graph
        word_count = len(query.split())
        has_question_word = any(
            query.lower().startswith(w)
            for w in ["what", "who", "when", "where", "why", "how", "which", "is", "are", "can", "does", "did"]
        )

        if has_question_word or word_count > 5:
            return ClassifyResult(
                category=QueryCategory.CORPUS_OPTIONAL,
                needs_corpus=True,
                needs_graph=False,
                needs_gatherer=True,
                confidence=0.6,
                reasoning="Question format detected, corpus likely helpful",
            )

        # Declarative or command-like, may not need corpus
        return ClassifyResult(
            category=QueryCategory.CORPUS_OPTIONAL,
            needs_corpus=True,  # Default to yes since we're a RAG system
            needs_graph=False,
            needs_gatherer=True,
            confidence=0.5,
            reasoning="Ambiguous query, defaulting to corpus retrieval",
        )


# Global instance
_corpus_classifier: CorpusClassifier | None = None


def get_corpus_classifier() -> CorpusClassifier:
    """Get or create global corpus classifier."""
    global _corpus_classifier
    if _corpus_classifier is None:
        _corpus_classifier = CorpusClassifier()
    return _corpus_classifier


__all__ = [
    "QueryCategory",
    "ClassifyResult",
    "CorpusClassifier",
    "get_corpus_classifier",
]
