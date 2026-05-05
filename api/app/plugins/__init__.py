"""Plugin architecture for extensible RAG components.

This module provides stable interfaces (ABCs) for all pluggable components:
- Ingestors: Parse different file formats
- Chunkers: Split text into chunks
- Embedders: Generate embeddings
- Retrievers: Find relevant documents
- Rerankers: Re-score retrieved documents
- Compressors: Reduce context size
- PostProcessors: Modify final output

The goal is to make "add a new retriever" a 30-minute task, not a refactor.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Core Types
# =============================================================================

@dataclass
class Chunk:
    """A text chunk with metadata."""
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "has_embedding": self.embedding is not None,
        }


@dataclass
class RetrievalResult:
    """Result from a retriever."""
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class ChunkConfig:
    """Configuration for chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separator: str = "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separator": self.separator,
        }


@dataclass
class ProcessContext:
    """Context passed to post-processors."""
    query: str
    chunks: list[Chunk]
    answer: str
    trace_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Plugin Base
# =============================================================================

class PluginType(Enum):
    """Types of plugins."""
    INGESTOR = "ingestor"
    CHUNKER = "chunker"
    EMBEDDER = "embedder"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    COMPRESSOR = "compressor"
    POST_PROCESSOR = "post_processor"


@dataclass
class PluginInfo:
    """Metadata about a plugin."""
    name: str
    plugin_type: PluginType
    version: str
    description: str
    author: str = ""
    config_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.plugin_type.value,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "config_schema": self.config_schema,
        }


class Plugin(ABC):
    """Base class for all plugins."""

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Return plugin metadata."""
        pass

    def configure(self, config: dict[str, Any]) -> None:  # noqa: B027
        """Configure the plugin. Override if needed."""
        pass

    def health_check(self) -> bool:
        """Check if plugin is healthy. Override if needed."""
        return True


# =============================================================================
# Ingestor Plugin
# =============================================================================

class IngestorPlugin(Plugin):
    """Plugin for ingesting documents of different formats."""

    @abstractmethod
    def can_handle(self, file_path: Path, content_type: str | None = None) -> bool:
        """Check if this ingestor can handle the file.

        Args:
            file_path: Path to the file
            content_type: MIME type if known

        Returns:
            True if this ingestor can process the file
        """
        pass

    @abstractmethod
    def ingest(self, file_path: Path, content: bytes) -> str:
        """Extract text content from a file.

        Args:
            file_path: Path to the file (for metadata)
            content: Raw file bytes

        Returns:
            Extracted text content
        """
        pass

    @property
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        return []


# =============================================================================
# Chunker Plugin
# =============================================================================

class ChunkerPlugin(Plugin):
    """Plugin for splitting text into chunks."""

    @abstractmethod
    def chunk(self, text: str, config: ChunkConfig) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Input text to chunk
            config: Chunking configuration

        Returns:
            List of chunks
        """
        pass

    def estimate_chunk_count(self, text: str, config: ChunkConfig) -> int:
        """Estimate number of chunks without actually chunking."""
        return max(1, len(text) // config.chunk_size)


# =============================================================================
# Embedder Plugin
# =============================================================================

class EmbedderPlugin(Plugin):
    """Plugin for generating embeddings."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced."""
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    def embed_query(self, query: str) -> list[float]:
        """Embed a query. May use different processing than documents."""
        return self.embed([query])[0]

    @property
    def max_batch_size(self) -> int:
        """Maximum texts to embed in one batch."""
        return 100


# =============================================================================
# Retriever Plugin
# =============================================================================

class RetrieverPlugin(Plugin):
    """Plugin for retrieving relevant documents."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of retrieval results ordered by relevance
        """
        pass

    def index(self, chunks: list[Chunk]) -> None:
        """Index chunks for retrieval. Override if needed."""
        pass

    def clear_index(self) -> None:
        """Clear the index. Override if needed."""
        pass

    @property
    def requires_embeddings(self) -> bool:
        """Whether this retriever needs pre-computed embeddings."""
        return False


# =============================================================================
# Reranker Plugin
# =============================================================================

class RerankerPlugin(Plugin):
    """Plugin for re-ranking retrieved documents."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Re-rank retrieval results.

        Args:
            query: Original query
            results: Initial retrieval results

        Returns:
            Re-ranked results with updated scores
        """
        pass

    @property
    def max_passages(self) -> int:
        """Maximum passages to rerank at once."""
        return 100


# =============================================================================
# Compressor Plugin
# =============================================================================

class CompressorPlugin(Plugin):
    """Plugin for compressing context to reduce tokens."""

    @abstractmethod
    def compress(
        self,
        query: str,
        chunks: list[Chunk],
        max_tokens: int,
    ) -> list[Chunk]:
        """Compress context to fit within token budget.

        Args:
            query: Query for relevance-aware compression
            chunks: Chunks to compress
            max_tokens: Maximum tokens in output

        Returns:
            Compressed chunks
        """
        pass

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count. Override for accuracy."""
        return len(text) // 4  # Rough estimate


# =============================================================================
# Post-Processor Plugin
# =============================================================================

class PostProcessorPlugin(Plugin):
    """Plugin for post-processing answers."""

    @abstractmethod
    def process(self, context: ProcessContext) -> str:
        """Process the answer.

        Args:
            context: Full context including query, chunks, and answer

        Returns:
            Processed answer
        """
        pass

    @property
    def order(self) -> int:
        """Execution order. Lower = earlier. Default 100."""
        return 100


# =============================================================================
# Plugin Registry
# =============================================================================

class PluginRegistry:
    """Registry for discovering and managing plugins."""

    def __init__(self) -> None:
        self._plugins: dict[PluginType, dict[str, Plugin]] = {
            pt: {} for pt in PluginType
        }
        self._plugin_paths: list[Path] = []

    def register(self, plugin: Plugin) -> None:
        """Register a plugin instance.

        Args:
            plugin: Plugin instance to register
        """
        info = plugin.info
        self._plugins[info.plugin_type][info.name] = plugin
        logger.info(f"Registered plugin: {info.name} ({info.plugin_type.value})")

    def unregister(self, plugin_type: PluginType, name: str) -> bool:
        """Unregister a plugin.

        Returns True if plugin was found and removed.
        """
        if name in self._plugins[plugin_type]:
            del self._plugins[plugin_type][name]
            return True
        return False

    def get(self, plugin_type: PluginType, name: str) -> Plugin | None:
        """Get a plugin by type and name."""
        return self._plugins[plugin_type].get(name)

    def get_all(self, plugin_type: PluginType) -> dict[str, Plugin]:
        """Get all plugins of a type."""
        return self._plugins[plugin_type].copy()

    def list_all(self) -> list[PluginInfo]:
        """List info for all registered plugins."""
        result = []
        for plugins in self._plugins.values():
            for plugin in plugins.values():
                result.append(plugin.info)
        return result

    def discover(self, path: Path) -> int:
        """Auto-discover plugins from a directory.

        Looks for Python files with a `create_plugin()` factory function.

        Args:
            path: Directory to search for plugins

        Returns:
            Number of plugins discovered
        """
        if not path.exists() or not path.is_dir():
            return 0

        discovered = 0
        self._plugin_paths.append(path)

        for py_file in path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    py_file.stem, py_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Look for factory function
                    if hasattr(module, "create_plugin"):
                        plugin = module.create_plugin()
                        if isinstance(plugin, Plugin):
                            self.register(plugin)
                            discovered += 1
                        elif isinstance(plugin, list):
                            for p in plugin:
                                if isinstance(p, Plugin):
                                    self.register(p)
                                    discovered += 1
            except Exception as e:
                logger.warning(f"Failed to load plugin from {py_file}: {e}")

        return discovered

    def configure_all(self, config: dict[str, dict[str, Any]]) -> None:
        """Configure all plugins from a config dict.

        Args:
            config: Dict mapping plugin names to their configs
        """
        for plugins in self._plugins.values():
            for name, plugin in plugins.items():
                if name in config:
                    plugin.configure(config[name])

    def health_check_all(self) -> dict[str, bool]:
        """Check health of all plugins.

        Returns dict mapping plugin names to health status.
        """
        results = {}
        for plugins in self._plugins.values():
            for name, plugin in plugins.items():
                try:
                    results[name] = plugin.health_check()
                except Exception:
                    results[name] = False
        return results


# =============================================================================
# Singleton Registry
# =============================================================================

_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


# =============================================================================
# Convenience Functions
# =============================================================================

def get_ingestor(name: str) -> IngestorPlugin | None:
    """Get an ingestor plugin by name."""
    plugin = get_plugin_registry().get(PluginType.INGESTOR, name)
    return plugin if isinstance(plugin, IngestorPlugin) else None


def get_chunker(name: str) -> ChunkerPlugin | None:
    """Get a chunker plugin by name."""
    plugin = get_plugin_registry().get(PluginType.CHUNKER, name)
    return plugin if isinstance(plugin, ChunkerPlugin) else None


def get_embedder(name: str) -> EmbedderPlugin | None:
    """Get an embedder plugin by name."""
    plugin = get_plugin_registry().get(PluginType.EMBEDDER, name)
    return plugin if isinstance(plugin, EmbedderPlugin) else None


def get_retriever(name: str) -> RetrieverPlugin | None:
    """Get a retriever plugin by name."""
    plugin = get_plugin_registry().get(PluginType.RETRIEVER, name)
    return plugin if isinstance(plugin, RetrieverPlugin) else None


def get_reranker(name: str) -> RerankerPlugin | None:
    """Get a reranker plugin by name."""
    plugin = get_plugin_registry().get(PluginType.RERANKER, name)
    return plugin if isinstance(plugin, RerankerPlugin) else None


__all__ = [
    # Types
    "Chunk",
    "RetrievalResult",
    "ChunkConfig",
    "ProcessContext",
    "PluginType",
    "PluginInfo",
    # Base
    "Plugin",
    # Plugin ABCs
    "IngestorPlugin",
    "ChunkerPlugin",
    "EmbedderPlugin",
    "RetrieverPlugin",
    "RerankerPlugin",
    "CompressorPlugin",
    "PostProcessorPlugin",
    # Registry
    "PluginRegistry",
    "get_plugin_registry",
    # Convenience
    "get_ingestor",
    "get_chunker",
    "get_embedder",
    "get_retriever",
    "get_reranker",
]
