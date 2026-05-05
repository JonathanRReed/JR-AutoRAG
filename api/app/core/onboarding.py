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
    pass

logger = logging.getLogger("autorag.onboarding")


SAMPLE_DOCUMENTS = [
    {
        "title": "JR AutoRAG Evaluation Brief",
        "content": """JR AutoRAG is a local-first Retrieval-Augmented Generation workbench for document-grounded AI systems.

The project is built to show several production concerns at once:
1. Private document ingestion and parsing
2. Hybrid retrieval with dense vectors, sparse matching, and reranking
3. Answer generation with citations, traces, and quality signals
4. Local provider support for Ollama and LM Studio
5. Security controls for client-adjacent work

Evaluator takeaway:
JR AutoRAG should be judged as a working RAG product, not a notebook demo. The strongest demo path is upload or seed documents, ask a grounded question, inspect cited evidence, then inspect the trace and quality cockpit.
""",
        "tags": ["demo", "overview", "evaluation"],
        "demo_question": "What should an evaluator notice first about JR AutoRAG?",
    },
    {
        "title": "State of the Art RAG Playbook",
        "content": """Modern RAG systems are moving beyond fixed top-k retrieval.

Important research directions:
1. Self-RAG adds retrieval and critique decisions so a model can reflect on evidence quality.
2. Corrective RAG adds a retrieval evaluator and recovery actions when evidence quality is weak.
3. Adaptive-RAG routes simple and complex questions through different retrieval strategies.
4. DRAGIN retrieves dynamically when the model has new information needs.
5. RAPTOR retrieves from hierarchical summaries to improve long-document reasoning.
6. GraphRAG and LightRAG add graph structure for relationship and corpus-level questions.

JR AutoRAG maps these ideas into product features through routing, HyDE, hybrid retrieval, reranking, self critique, evidence contracts, citation verification, RAPTOR, and GraphRAG hooks.
""",
        "tags": ["demo", "research", "rag"],
        "demo_question": "Which current RAG research ideas does this project already surface?",
    },
    {
        "title": "Peer Product Comparison",
        "content": """Open-source RAG peers set clear expectations for product quality.

Kotaemon is strong at citation preview, hybrid retrieval, reranking, and low-relevance warnings.
RAGFlow is strong at deep document understanding, OCR, parsing, and agent templates.
AnythingLLM is strong at local-first setup and fast onboarding.
R2R is strong at REST APIs, multimodal ingestion, hybrid search, knowledge graphs, and production-facing retrieval.
Haystack is strong at modular pipelines, branching, loops, and deployment patterns.
AutoRAG is strong at evaluation-driven pipeline optimization.

JR AutoRAG should compete by combining local-first operation, visible evidence, transparent traces, and a guided demo that reaches first answer quickly.
""",
        "tags": ["demo", "peers", "product"],
        "demo_question": "How does JR AutoRAG compare with open-source RAG peers?",
    },
    {
        "title": "Project Manager Demo Scenario",
        "content": """A project manager evaluating JR AutoRAG should see a clear workflow.

Demo script:
1. Confirm the API is connected.
2. Seed the demo corpus.
3. Ask what the system is and why it matters.
4. Inspect the answer citations.
5. Open the pipeline trace and explain each retrieval stage.
6. Open the quality cockpit and review extraction, recommendations, and advisory experiments.
7. Explain that data can be disposable in demo mode and local-first in real client use.

Success criteria:
The product should make evidence, risk, readiness, and next actions obvious without requiring the evaluator to read source code.
""",
        "tags": ["demo", "project-management", "workflow"],
        "demo_question": "Give me a project-manager style demo script for this app.",
    },
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
        "demo_question": "What is RAG and what are its benefits?",
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
        "demo_question": "What chunk size should I use for technical documents?",
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
        "demo_question": "Compare hybrid search and RAPTOR.",
    },
]


EXAMPLE_QUERIES = [
    {
        "query": "What should an evaluator notice first about JR AutoRAG?",
        "category": "demo",
        "expected_docs": ["JR AutoRAG Evaluation Brief"],
    },
    {
        "query": "Which current RAG research ideas does this project already surface?",
        "category": "research",
        "expected_docs": ["State of the Art RAG Playbook"],
    },
    {
        "query": "Give me a project-manager style demo script for this app.",
        "category": "workflow",
        "expected_docs": ["Project Manager Demo Scenario"],
    },
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
    def create_default(cls) -> OnboardingFlow:
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
