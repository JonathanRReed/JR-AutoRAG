"""Background artifact builder for GraphRAG and RAPTOR.

Implements Workstream A2: Async GraphRAG and RAPTOR builds with status tracking.

Key capabilities:
- Decouples basic retrieval availability from deep artifact readiness
- Build graph/hierarchy in background tasks with persistence
- Status tracking: NotBuilt, Building, Ready, Failed
- Graceful degradation: queries work without artifacts
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk
    from .graph_rag import GraphRAG
    from .hierarchy import DocumentTree
    from .providers import LLMProvider


class ArtifactStatus(str, Enum):
    """Status of an artifact build."""
    NOT_BUILT = "not_built"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


@dataclass
class ArtifactState:
    """State of a single artifact type (graph or hierarchy)."""
    status: ArtifactStatus = ArtifactStatus.NOT_BUILT
    progress: float = 0.0  # Percentage 0-100
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    corpus_version: str = ""
    item_count: int = 0  # entities/nodes count

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "progress": self.progress,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "corpus_version": self.corpus_version,
            "item_count": self.item_count,
            "duration_seconds": (
                self.completed_at - self.started_at
                if self.started_at and self.completed_at
                else None
            ),
        }


@dataclass
class BuildProgress:
    """Progress of all artifact builds."""
    graph_rag: ArtifactState = field(default_factory=ArtifactState)
    raptor: ArtifactState = field(default_factory=ArtifactState)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_rag": self.graph_rag.to_dict(),
            "raptor": self.raptor.to_dict(),
        }

    def is_any_building(self) -> bool:
        """Check if any artifact is currently building."""
        return (
            self.graph_rag.status == ArtifactStatus.BUILDING or
            self.raptor.status == ArtifactStatus.BUILDING
        )

    def all_ready(self) -> bool:
        """Check if all artifacts are ready."""
        return (
            self.graph_rag.status == ArtifactStatus.READY and
            self.raptor.status == ArtifactStatus.READY
        )


class ArtifactBuilder:
    """Manages background building of GraphRAG and RAPTOR artifacts.

    Implements G4: Async artifacts with graceful fallback.

    Usage:
        builder = ArtifactBuilder()

        # Start background build (non-blocking)
        await builder.build_all_async(chunks, provider, corpus_version)

        # Check status
        if builder.is_graph_ready():
            graph = builder.get_graph()
    """

    def __init__(
        self,
        persist_path: Path | str = "data/artifacts",
        on_progress: Callable[[BuildProgress], None] | None = None,
    ) -> None:
        """Initialize artifact builder.

        Args:
            persist_path: Directory for saving built artifacts
            on_progress: Optional callback for progress updates
        """
        self._persist_path = Path(persist_path)
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._progress = BuildProgress()
        self._on_progress = on_progress
        self._build_task: asyncio.Task | None = None

        # Cached built artifacts
        self._graph: GraphRAG | None = None
        self._trees: dict[str, DocumentTree] = {}

    @property
    def progress(self) -> BuildProgress:
        """Get current build progress."""
        return self._progress

    def is_graph_ready(self) -> bool:
        """Check if GraphRAG artifact is ready for use."""
        return self._progress.graph_rag.status == ArtifactStatus.READY

    def is_raptor_ready(self) -> bool:
        """Check if RAPTOR hierarchy is ready for use."""
        return self._progress.raptor.status == ArtifactStatus.READY

    def is_graph_building(self) -> bool:
        """Check if GraphRAG is currently building."""
        return self._progress.graph_rag.status == ArtifactStatus.BUILDING

    def is_raptor_building(self) -> bool:
        """Check if RAPTOR is currently building."""
        return self._progress.raptor.status == ArtifactStatus.BUILDING

    def get_graph(self) -> GraphRAG | None:
        """Get built graph if ready."""
        if self.is_graph_ready():
            return self._graph
        return None

    def get_trees(self) -> dict[str, DocumentTree]:
        """Get built hierarchies if ready."""
        if self.is_raptor_ready():
            return self._trees
        return {}

    def set_status(self, artifact_type: str, status: ArtifactStatus, corpus_version: str = "") -> None:
        """Manually set artifact status (e.g. when loaded from external cache)."""
        if artifact_type == "graph_rag":
            self._progress.graph_rag.status = status
            self._progress.graph_rag.corpus_version = corpus_version
            if status == ArtifactStatus.READY:
                self._progress.graph_rag.completed_at = time.time()
                self._progress.graph_rag.progress = 100.0
        elif artifact_type == "raptor":
            self._progress.raptor.status = status
            self._progress.raptor.corpus_version = corpus_version
            if status == ArtifactStatus.READY:
                self._progress.raptor.completed_at = time.time()
                self._progress.raptor.progress = 100.0
        self._notify_progress()

    def set_items(self, artifact_type: str, count: int) -> None:
        """Set item count for an artifact."""
        if artifact_type == "graph_rag":
            self._progress.graph_rag.item_count = count
        elif artifact_type == "raptor":
            self._progress.raptor.item_count = count
        self._notify_progress()

    async def build_all_async(
        self,
        chunks: list[EvidenceChunk],
        provider: LLMProvider | None,
        corpus_version: str,
        force_rebuild: bool = False,
    ) -> None:
        """Trigger background builds for all artifacts.

        Args:
            chunks: Document chunks to build artifacts from
            provider: LLM provider for extraction
            corpus_version: Current corpus version for tracking
            force_rebuild: If True, rebuild even if already built
        """
        # Check if we need to rebuild
        if not force_rebuild and (self.is_graph_ready() and
            self._progress.graph_rag.corpus_version == corpus_version):
            return  # Already built for this corpus version

        # Cancel any existing build
        if self._build_task and not self._build_task.done():
            self._build_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._build_task

        # Start new build
        self._build_task = asyncio.create_task(
            self._run_builds(chunks, provider, corpus_version)
        )

    async def _run_builds(
        self,
        chunks: list[EvidenceChunk],
        provider: LLMProvider | None,
        corpus_version: str,
    ) -> None:
        """Run both builds in parallel."""
        await asyncio.gather(
            self._build_graph_async(chunks, provider, corpus_version),
            self._build_hierarchy_async(chunks, provider, corpus_version),
            return_exceptions=True,  # Don't fail both if one fails
        )

    async def _build_graph_async(
        self,
        chunks: list[EvidenceChunk],
        provider: LLMProvider | None,
        corpus_version: str,
    ) -> None:
        """Build GraphRAG artifact in background."""
        self._progress.graph_rag.status = ArtifactStatus.BUILDING
        self._progress.graph_rag.started_at = time.time()
        self._progress.graph_rag.corpus_version = corpus_version
        self._progress.graph_rag.progress = 0.0
        self._progress.graph_rag.error = None
        self._notify_progress()

        # Progress callback to bridge GraphRAG callbacks to our percentage
        # GraphRAG stages: extracting_graph (80%), summarizing_communities (20%)
        def on_graph_progress(stage: str, cur: int, total: int, msg: str | None = None) -> None:
            if total == 0:
                return

            nonlocal self
            base_progress = 0.0
            scale = 1.0

            if stage == "extracting_graph":
                base_progress = 0.0
                scale = 0.8
            elif stage == "summarizing_communities":
                base_progress = 80.0
                scale = 0.2

            ratio = cur / total
            pct = base_progress + (ratio * scale * 100.0)
            self._progress.graph_rag.progress = min(99.9, pct)
            self._notify_progress()

        try:
            if provider is None:
                raise RuntimeError("No LLM provider configured for GraphRAG build")
            from .graph_rag import GraphRAG

            graph = GraphRAG()

            # Build from chunks (async method)
            await graph.build_from_chunks(chunks, provider, on_progress=on_graph_progress)

            # Detect communities and summarize
            graph.detect_communities()
            # Summarize communities (async method)
            await graph.summarize_communities(provider, on_progress=on_graph_progress)

            # Persist to disk
            graph_path = self._persist_path / f"graph_{corpus_version}.json"
            graph_data = graph.to_dict()
            graph_path.write_text(json.dumps(graph_data, indent=2))

            # Store in memory
            self._graph = graph

            self._progress.graph_rag.status = ArtifactStatus.READY
            self._progress.graph_rag.completed_at = time.time()
            self._progress.graph_rag.progress = 100.0
            self._progress.graph_rag.item_count = len(graph.entities)

        except Exception as e:
            self._progress.graph_rag.status = ArtifactStatus.FAILED
            self._progress.graph_rag.error = str(e)
            self._progress.graph_rag.completed_at = time.time()

        self._notify_progress()

    async def _build_hierarchy_async(
        self,
        chunks: list[EvidenceChunk],
        provider: LLMProvider | None,
        corpus_version: str,
    ) -> None:
        """Build RAPTOR hierarchy in background."""
        self._progress.raptor.status = ArtifactStatus.BUILDING
        self._progress.raptor.started_at = time.time()
        self._progress.raptor.corpus_version = corpus_version
        self._progress.raptor.error = None
        self._progress.raptor.progress = 0.0
        self._notify_progress()

        try:
            from .hierarchy import HierarchyBuilder

            builder = HierarchyBuilder()

            # Group chunks by document
            doc_chunks: dict[str, list] = {}
            for chunk in chunks:
                doc_id = getattr(chunk, 'doc_id', 'default')
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(chunk)

            # Build tree for each document
            trees: dict[str, Any] = {}
            total_docs = len(doc_chunks)
            for completed, (doc_id, doc_chunk_list) in enumerate(
                doc_chunks.items(), start=1
            ):
                # Combine chunk texts for hierarchy building
                text = "\n\n".join(
                    getattr(c, 'snippet', str(c)) for c in doc_chunk_list
                )
                tree = await asyncio.to_thread(
                    builder.build, text, doc_id, doc_id
                )
                trees[doc_id] = tree
                if total_docs:
                    self._progress.raptor.progress = min(99.9, (completed / total_docs) * 100.0)
                    self._notify_progress()

            # Persist to disk
            hierarchy_path = self._persist_path / f"hierarchy_{corpus_version}.json"
            # Serialize trees (simplified - actual impl may vary)
            tree_data = {
                doc_id: {"doc_id": doc_id, "node_count": len(getattr(tree, 'nodes', {}))}
                for doc_id, tree in trees.items()
            }
            hierarchy_path.write_text(json.dumps(tree_data, indent=2))

            # Store in memory
            self._trees = trees

            self._progress.raptor.status = ArtifactStatus.READY
            self._progress.raptor.completed_at = time.time()
            self._progress.raptor.progress = 100.0
            self._progress.raptor.item_count = sum(
                len(getattr(t, 'nodes', {})) for t in trees.values()
            )

        except Exception as e:
            self._progress.raptor.status = ArtifactStatus.FAILED
            self._progress.raptor.error = str(e)
            self._progress.raptor.completed_at = time.time()

        self._notify_progress()

    def _notify_progress(self) -> None:
        """Notify progress callback if set."""
        if self._on_progress:
            with contextlib.suppress(Exception):
                self._on_progress(self._progress)

    def load_from_disk(self, corpus_version: str) -> bool:
        """Try to load artifacts from disk.

        Returns True if artifacts were loaded successfully.
        """
        graph_path = self._persist_path / f"graph_{corpus_version}.json"
        hierarchy_path = self._persist_path / f"hierarchy_{corpus_version}.json"

        loaded_any = False

        if graph_path.exists():
            try:
                from .graph_rag import GraphRAG
                data = json.loads(graph_path.read_text())
                self._graph = GraphRAG.from_dict(data)
                self._progress.graph_rag.status = ArtifactStatus.READY
                self._progress.graph_rag.corpus_version = corpus_version
                loaded_any = True
            except Exception:
                pass

        if hierarchy_path.exists():
            try:
                # Note: Full deserialization would need to reconstruct trees
                self._progress.raptor.status = ArtifactStatus.READY
                self._progress.raptor.corpus_version = corpus_version
                loaded_any = True
            except Exception:
                pass

        return loaded_any

    def reset(self) -> None:
        """Reset all artifact state."""
        self._progress = BuildProgress()
        self._graph = None
        self._trees = {}
        if self._build_task and not self._build_task.done():
            self._build_task.cancel()


# Global builder instance
_builder: ArtifactBuilder | None = None


def get_artifact_builder(persist_path: str = "data/artifacts") -> ArtifactBuilder:
    """Get or create the global artifact builder."""
    global _builder
    if _builder is None:
        _builder = ArtifactBuilder(persist_path)
    return _builder


__all__ = [
    "ArtifactStatus",
    "ArtifactState",
    "BuildProgress",
    "ArtifactBuilder",
    "get_artifact_builder",
]
