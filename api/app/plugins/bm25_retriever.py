"""Example BM25 retriever plugin.

This demonstrates how to create a new retriever plugin in ~30 minutes.
Use this as a template for your own retrievers.
"""

from __future__ import annotations

import re
from typing import Any

from . import (
    Chunk,
    Plugin,
    PluginInfo,
    PluginType,
    RetrievalResult,
    RetrieverPlugin,
)


class BM25RetrieverPlugin(RetrieverPlugin):
    """Simple BM25 (Best Matching 25) retriever.

    This is a keyword-based retriever that uses term frequency and
    inverse document frequency for scoring.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._avg_len: float = 0.0

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="bm25",
            plugin_type=PluginType.RETRIEVER,
            version="1.0.0",
            description="BM25 keyword-based retriever",
            author="JR AutoRAG",
            config_schema={
                "k1": {"type": "number", "default": 1.5},
                "b": {"type": "number", "default": 0.75},
            },
        )

    def configure(self, config: dict[str, Any]) -> None:
        """Update configuration."""
        if "k1" in config:
            self.k1 = float(config["k1"])
        if "b" in config:
            self.b = float(config["b"])

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        return re.findall(r"\w+", text.lower())

    def index(self, chunks: list[Chunk]) -> None:
        """Index chunks for BM25 retrieval."""
        self._chunks = chunks
        self._tokenized = [self._tokenize(c.text) for c in chunks]

        # Compute document frequencies
        self._doc_freqs = {}
        for tokens in self._tokenized:
            seen = set()
            for token in tokens:
                if token not in seen:
                    self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                    seen.add(token)

        # Compute average document length
        if self._tokenized:
            self._avg_len = sum(len(t) for t in self._tokenized) / len(self._tokenized)
        else:
            self._avg_len = 0.0

    def clear_index(self) -> None:
        """Clear the index."""
        self._chunks = []
        self._tokenized = []
        self._doc_freqs = {}
        self._avg_len = 0.0

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Retrieve documents using BM25 scoring."""
        if not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        n = len(self._chunks)

        scores: list[tuple[int, float]] = []

        for idx, doc_tokens in enumerate(self._tokenized):
            doc_len = len(doc_tokens)
            score = 0.0

            # Count term frequencies in document
            tf = {}
            for token in doc_tokens:
                tf[token] = tf.get(token, 0) + 1

            for token in query_tokens:
                if token not in self._doc_freqs:
                    continue

                # IDF component
                df = self._doc_freqs[token]
                idf = (n - df + 0.5) / (df + 0.5)
                if idf > 0:
                    idf = idf + 1.0  # Log isn't needed, this is valid BM25 variant

                # TF component with length normalization
                freq = tf.get(token, 0)
                if freq > 0 and self._avg_len > 0:
                    tf_norm = (freq * (self.k1 + 1)) / (
                        freq + self.k1 * (1 - self.b + self.b * doc_len / self._avg_len)
                    )
                    score += idf * tf_norm

            if score > 0:
                scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top k
        results = []
        for idx, score in scores[:k]:
            chunk = self._chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    score=score,
                    metadata=chunk.metadata,
                )
            )

        return results

    @property
    def requires_embeddings(self) -> bool:
        """BM25 doesn't need embeddings."""
        return False


# =============================================================================
# Factory Function (required for auto-discovery)
# =============================================================================


def create_plugin() -> Plugin:
    """Factory function for auto-discovery."""
    return BM25RetrieverPlugin()
