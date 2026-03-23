"""Conversation memory for multi-turn RAG interactions.

This module provides:
- Turn-by-turn conversation storage
- Context accumulation across turns
- Relevant history retrieval for follow-up queries
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional fields for RAG context
    chunks_used: list[str] = field(default_factory=list)  # Chunk IDs
    query_type: str = ""


@dataclass
class ConversationContext:
    """Context extracted from conversation history."""
    relevant_turns: list[ConversationTurn]
    summary: str
    key_entities: list[str]
    follow_up_detected: bool


@dataclass
class EpisodicMemory:
    id: str
    summary: str
    timestamp: datetime
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """Stores and manages conversation history for multi-turn RAG.

    Features:
    - Automatic turn storage
    - Follow-up query detection
    - Context window management
    - Entity extraction for context
    """

    # Patterns indicating follow-up questions
    FOLLOW_UP_PATTERNS = [
        r"^(what|how|why|when|where|who) (about|else|more)\b",
        r"^(and|also|additionally)\b",
        r"^(but|however)\b",
        r"^(can you|could you) (explain|elaborate|tell me more)",
        r"^(it|that|this|they|those)\b",
        r"\b(previous|earlier|above|before)\b",
        r"\b(you (said|mentioned|told))\b",
    ]

    # Words that suggest a new topic
    NEW_TOPIC_PATTERNS = [
        r"^(let's talk about|changing topic|new question)\b",
        r"^(forget|ignore) (that|previous)\b",
        r"^(start over|reset)\b",
    ]

    def __init__(
        self,
        max_turns: int = 20,
        context_window_turns: int = 5,
    ) -> None:
        self._max_turns = max_turns
        self._context_window = context_window_turns
        self._conversations: dict[str, list[ConversationTurn]] = {}
        self._episodic_memories: dict[str, list[EpisodicMemory]] = {}
        self._follow_up_re = [
            re.compile(p, re.IGNORECASE) for p in self.FOLLOW_UP_PATTERNS
        ]
        self._new_topic_re = [
            re.compile(p, re.IGNORECASE) for p in self.NEW_TOPIC_PATTERNS
        ]

    def create_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        conv_id = str(uuid.uuid4())
        self._conversations[conv_id] = []
        return conv_id

    def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        chunks_used: list[str] | None = None,
        query_type: str = "",
    ) -> ConversationTurn:
        """Add a turn to a conversation."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []

        turn = ConversationTurn(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            timestamp=datetime.now(UTC),
            metadata=metadata or {},
            chunks_used=chunks_used or [],
            query_type=query_type,
        )

        self._conversations[conversation_id].append(turn)

        # Trim if exceeds max
        if len(self._conversations[conversation_id]) > self._max_turns:
            self._conversations[conversation_id] = (
                self._conversations[conversation_id][-self._max_turns:]
            )

        return turn

    def get_turns(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[ConversationTurn]:
        """Get conversation turns, optionally limited."""
        turns = self._conversations.get(conversation_id, [])
        if limit:
            return turns[-limit:]
        return turns

    def is_follow_up(self, query: str) -> bool:
        """Detect if a query is a follow-up to previous context."""
        # Check for new topic indicators first
        for pattern in self._new_topic_re:
            if pattern.search(query):
                return False

        # Check for follow-up indicators
        for pattern in self._follow_up_re:
            if pattern.search(query):
                return True

        # Short queries are often follow-ups
        return len(query.split()) <= 5

    def _extract_entities(self, text: str) -> list[str]:
        """Extract potential named entities from text."""
        # Simple heuristic: capitalized words not at sentence start
        # This is a simplified version - production would use NER
        sentences = re.split(r'[.!?]\s+', text)
        entities = set()

        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                # Skip first word of sentence
                if i == 0:
                    continue
                # Check if capitalized (potential entity)
                if word and word[0].isupper() and len(word) > 2:
                    clean = re.sub(r'[^\w]', '', word)
                    if clean:
                        entities.add(clean)

        return list(entities)

    def get_context(
        self,
        conversation_id: str,
        current_query: str,
    ) -> ConversationContext:
        """Get relevant context for the current query."""
        turns = self.get_turns(conversation_id, limit=self._context_window)

        if not turns:
            return ConversationContext(
                relevant_turns=[],
                summary="",
                key_entities=[],
                follow_up_detected=False,
            )

        follow_up = self.is_follow_up(current_query)

        # Extract entities from recent conversation
        all_text = " ".join(t.content for t in turns)
        entities = self._extract_entities(all_text)

        # Build summary from last few turns
        recent = turns[-3:] if len(turns) >= 3 else turns
        summary_parts = []
        for turn in recent:
            role_label = "User" if turn.role == "user" else "Assistant"
            # Truncate long content
            content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
            summary_parts.append(f"{role_label}: {content}")
        summary = "\n".join(summary_parts)

        return ConversationContext(
            relevant_turns=turns,
            summary=summary,
            key_entities=entities[:10],  # Limit entities
            follow_up_detected=follow_up,
        )

    def build_context_prompt(
        self,
        conversation_id: str,
        current_query: str,
    ) -> str:
        """Build a context string for the LLM prompt."""
        context = self.get_context(conversation_id, current_query)
        episodic = self._episodic_memories.get(conversation_id, [])

        if not context.relevant_turns:
            if not episodic:
                return ""
            return "\n".join(
                [
                    "Important prior memories:",
                    *[f"- {memory.summary}" for memory in episodic[-3:]],
                ]
            )

        parts = ["Previous conversation:"]
        for turn in context.relevant_turns[-3:]:
            role = "User" if turn.role == "user" else "Assistant"
            content = turn.content[:300] + "..." if len(turn.content) > 300 else turn.content
            parts.append(f"{role}: {content}")
        if episodic:
            parts.append("Important prior memories:")
            parts.extend(f"- {memory.summary}" for memory in episodic[-3:])

        return "\n".join(parts)

    def score_memory_worthiness(
        self,
        user_query: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        score = 0.0
        if len(user_query.split()) >= 6:
            score += 0.25
        if len(answer.split()) >= 40:
            score += 0.2
        if metadata and metadata.get("chunks_used"):
            score += 0.2
        if metadata and metadata.get("sources_count", 0) >= 2:
            score += 0.2
        if self.is_follow_up(user_query):
            score += 0.1
        if any(term in user_query.lower() for term in ("remember", "preference", "always", "never", "my ", "our ")):
            score += 0.25
        return min(score, 1.0)

    def record_exchange(
        self,
        conversation_id: str,
        user_query: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_turn = self.add_turn(conversation_id, "user", user_query, metadata=metadata or {})
        assistant_turn = self.add_turn(
            conversation_id,
            "assistant",
            answer,
            metadata=metadata or {},
            chunks_used=list((metadata or {}).get("chunks_used", [])),
            query_type=str((metadata or {}).get("query_type", "")),
        )
        score = self.score_memory_worthiness(user_query, answer, metadata)
        memory_written = False
        if score >= 0.55:
            summary = self._summarize_exchange(user_query, answer)
            self._episodic_memories.setdefault(conversation_id, []).append(
                EpisodicMemory(
                    id=str(uuid.uuid4()),
                    summary=summary,
                    timestamp=datetime.now(UTC),
                    evidence=list((metadata or {}).get("chunks_used", [])),
                    metadata={"score": round(score, 3)},
                )
            )
            self._episodic_memories[conversation_id] = self._episodic_memories[conversation_id][-10:]
            memory_written = True
        return {
            "user_turn_id": user_turn.id,
            "assistant_turn_id": assistant_turn.id,
            "memory_score": round(score, 3),
            "memory_written": memory_written,
            "episodic_count": len(self._episodic_memories.get(conversation_id, [])),
        }

    def _summarize_exchange(self, user_query: str, answer: str) -> str:
        trimmed_query = user_query.strip()
        trimmed_answer = answer.strip().replace("\n", " ")
        if len(trimmed_answer) > 180:
            trimmed_answer = trimmed_answer[:177].rstrip() + "..."
        return f"Q: {trimmed_query} A: {trimmed_answer}"

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear a conversation's history."""
        if conversation_id in self._conversations:
            self._conversations[conversation_id] = []

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation entirely."""
        self._conversations.pop(conversation_id, None)


__all__ = [
    "ConversationTurn",
    "ConversationContext",
    "ConversationMemory",
]
