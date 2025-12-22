"""Onboarding flow and sample data.

This module implements P1.11: Better Onboarding
- Getting started flow
- Sample datasets
- Example query showcase
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("autorag.onboarding")


SAMPLE_DOCUMENTS = [
    {
        "title": "Introduction to RAG",
        "content": """Retrieval-Augmented Generation (RAG) combines retrieval with language models.
        
When a user asks a question, RAG:
1. Retrieves relevant documents from a knowledge base
2. Provides those documents as context to an LLM
3. Generates an answer grounded in the retrieved evidence

Benefits:
- Reduces hallucination by grounding answers in real data
- Allows for up-to-date information without retraining
- Provides citations and traceability
""",
        "tags": ["tutorial", "rag", "basics"],
    },
    {
        "title": "Best Practices for Document Ingestion",
        "content": """Follow these best practices when ingesting documents:

1. **Chunk Size**: Use 256-512 tokens per chunk for most use cases
2. **Overlap**: Add 50-100 token overlap between chunks for context
3. **Metadata**: Preserve document titles, dates, and sections
4. **Formats**: Support PDF, DOCX, Markdown, and plain text
5. **Quality**: Remove headers, footers, and boilerplate

For technical documents, consider smaller chunks.
For narrative content, larger chunks work better.
""",
        "tags": ["tutorial", "ingestion", "best-practices"],
    },
    {
        "title": "Advanced Retrieval Techniques",
        "content": """This document covers advanced retrieval strategies:

**Hybrid Search**: Combine dense (vector) and sparse (BM25) retrieval.
Typically weighted 70% dense, 30% sparse.

**RAPTOR**: Build hierarchical summaries for multi-document reasoning.
Useful for "compare across all documents" queries.

**GraphRAG**: Entity-based knowledge graph for relationship queries.
Best for "how does X relate to Y" type questions.

**Reranking**: Use a cross-encoder to reorder initial results.
Improves precision at the cost of latency.
""",
        "tags": ["advanced", "retrieval", "techniques"],
    },
]


EXAMPLE_QUERIES = [
    {
        "query": "What is RAG and what are its benefits?",
        "category": "factual",
        "expected_docs": ["Introduction to RAG"],
    },
    {
        "query": "What chunk size should I use for technical documents?",
        "category": "specific",
        "expected_docs": ["Best Practices for Document Ingestion"],
    },
    {
        "query": "Compare hybrid search and RAPTOR",
        "category": "comparison",
        "expected_docs": ["Advanced Retrieval Techniques"],
    },
]


@dataclass
class OnboardingStep:
    """A single onboarding step."""
    
    id: str
    title: str
    description: str
    action: str
    completed: bool = False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "action": self.action,
            "completed": self.completed,
        }


@dataclass
class OnboardingFlow:
    """Complete onboarding flow state."""
    
    steps: list[OnboardingStep] = field(default_factory=list)
    current_step: int = 0
    
    @classmethod
    def create_default(cls) -> "OnboardingFlow":
        """Create default onboarding flow."""
        steps = [
            OnboardingStep(
                id="configure_provider",
                title="Configure LLM Provider",
                description="Set up Ollama, LM Studio, or OpenAI connection",
                action="open_settings",
            ),
            OnboardingStep(
                id="upload_documents",
                title="Upload Documents",
                description="Add your first documents to the corpus",
                action="open_upload",
            ),
            OnboardingStep(
                id="try_query",
                title="Ask Your First Question",
                description="Try a query to see RAG in action",
                action="focus_query",
            ),
            OnboardingStep(
                id="explore_trace",
                title="Explore the Pipeline",
                description="See how your query was processed",
                action="open_trace",
            ),
            OnboardingStep(
                id="enable_security",
                title="Enable Security (Optional)",
                description="Set AUTORAG_AUTH_ENABLED=true for production",
                action="open_docs",
            ),
            OnboardingStep(
                id="run_evaluation",
                title="Run Evaluation Gates (Optional)",
                description="Test quality with built-in benchmarks",
                action="open_eval",
            ),
        ]
        return cls(steps=steps)
    
    @property
    def is_complete(self) -> bool:
        return all(s.completed for s in self.steps)
    
    @property
    def progress_percent(self) -> float:
        if not self.steps:
            return 100.0
        completed = sum(1 for s in self.steps if s.completed)
        return (completed / len(self.steps)) * 100
    
    def complete_step(self, step_id: str) -> bool:
        """Mark a step as completed."""
        for step in self.steps:
            if step.id == step_id:
                step.completed = True
                # Move to next incomplete step
                for i, s in enumerate(self.steps):
                    if not s.completed:
                        self.current_step = i
                        return True
                self.current_step = len(self.steps)
                return True
        return False
    
    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "progress": round(self.progress_percent),
            "is_complete": self.is_complete,
        }


def get_sample_documents() -> list[dict]:
    """Get sample documents for demo corpus."""
    return SAMPLE_DOCUMENTS


def get_example_queries() -> list[dict]:
    """Get example queries for showcase."""
    return EXAMPLE_QUERIES


def create_onboarding_flow() -> OnboardingFlow:
    """Create new onboarding flow."""
    return OnboardingFlow.create_default()


__all__ = [
    "SAMPLE_DOCUMENTS",
    "EXAMPLE_QUERIES",
    "OnboardingStep",
    "OnboardingFlow",
    "get_sample_documents",
    "get_example_queries",
    "create_onboarding_flow",
]
