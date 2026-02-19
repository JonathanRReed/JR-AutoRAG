"""HyDE: Hypothetical Document Embeddings for enhanced retrieval.

This module implements the HyDE technique from the paper:
"Precise Zero-Shot Dense Retrieval without Relevance Labels"

HyDE generates a hypothetical document that would answer the query,
then embeds that document to find similar real documents.
This significantly improves zero-shot retrieval accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers import LLMProvider


@dataclass
class HyDEConfig:
    """Configuration for HyDE generation."""

    # Number of hypothetical documents to generate
    num_hypotheticals: int = 1

    # Maximum tokens for each hypothetical document
    max_tokens: int = 150

    # Temperature for generation (lower = more focused)
    temperature: float = 0.7

    # Whether to combine query with hypothetical for final embedding
    combine_with_query: bool = True

    # Document types for different query categories
    document_templates: dict[str, str] = field(default_factory=lambda: {
        "factual": "Write a short technical document excerpt that definitively answers this question: {query}",
        "procedural": "Write a step-by-step procedure or how-to guide excerpt that explains: {query}",
        "analytical": "Write an analytical report excerpt that provides detailed analysis for: {query}",
        "comparative": "Write a comparison document excerpt that compares and contrasts: {query}",
        "summary": "Write a comprehensive summary document that covers: {query}",
        "default": "Write a Wikipedia-style encyclopedia article excerpt that directly answers: {query}",
    })


@dataclass
class HyDEResult:
    """Result of HyDE generation."""

    # The query that was processed
    original_query: str

    # Generated hypothetical documents
    hypotheticals: list[str]

    # Final embedding text (may combine query + hypotheticals)
    embedding_text: str

    # Query type detected
    query_type: str | None = None


class HyDEGenerator:
    """Generates hypothetical documents for HyDE retrieval.

    Usage:
        hyde = HyDEGenerator()
        result = await hyde.generate(query, provider)
        # Use result.embedding_text for embedding
    """

    def __init__(self, config: HyDEConfig | None = None) -> None:
        self.config = config or HyDEConfig()

    async def generate(
        self,
        query: str,
        provider: LLMProvider,
        query_type: str | None = None,
    ) -> HyDEResult:
        """Generate hypothetical document(s) for the query.

        Args:
            query: The user's question
            provider: LLM provider for generation
            query_type: Optional query type for template selection

        Returns:
            HyDEResult with hypothetical documents and embedding text
        """
        # Select appropriate template
        template_key = query_type or "default"
        if template_key not in self.config.document_templates:
            template_key = "default"

        template = self.config.document_templates[template_key]
        prompt = template.format(query=query)

        hypotheticals = []

        for _ in range(self.config.num_hypotheticals):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a technical document generator. "
                            "Generate a realistic document excerpt that would contain the answer "
                            "to the user's question. Be specific and factual in your writing style. "
                            "Write 2-3 sentences that directly address the topic. "
                            "Do not include phrases like 'This document' or 'According to'. "
                            "Just write the content as if it's from a real document."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]

                hypothetical = await provider.chat(
                    messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )

                # Clean up the response
                hypothetical = hypothetical.strip()
                if hypothetical:
                    hypotheticals.append(hypothetical)

            except Exception:
                # If generation fails, continue with what we have
                continue

        # Build final embedding text
        if self.config.combine_with_query and hypotheticals:
            # Combine query with hypothetical for better matching
            embedding_text = f"{query}\n\n{hypotheticals[0]}"
        elif hypotheticals:
            embedding_text = hypotheticals[0]
        else:
            # Fallback to original query if no hypotheticals generated
            embedding_text = query

        return HyDEResult(
            original_query=query,
            hypotheticals=hypotheticals,
            embedding_text=embedding_text,
            query_type=query_type,
        )

    def generate_sync(
        self,
        query: str,
        provider: LLMProvider,
        query_type: str | None = None,
    ) -> HyDEResult:
        """Synchronous wrapper for generate."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context, can't use run()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.generate(query, provider, query_type)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.generate(query, provider, query_type)
                )
        except RuntimeError:
            return asyncio.run(self.generate(query, provider, query_type))


# Singleton for reuse
_hyde_generator: HyDEGenerator | None = None


def get_hyde_generator(config: HyDEConfig | None = None) -> HyDEGenerator:
    """Get or create a HyDE generator instance."""
    global _hyde_generator
    if _hyde_generator is None or config is not None:
        _hyde_generator = HyDEGenerator(config)
    return _hyde_generator


__all__ = [
    "HyDEConfig",
    "HyDEResult",
    "HyDEGenerator",
    "get_hyde_generator",
]
