"""Configuration snapshots for reproducibility.

Implements immutable config snapshots that capture the complete state of the
system at query time, enabling reproducible runs and trace replay.

Each snapshot includes:
- Model and embedding configuration
- Retrieval settings (dense_k, rerankers, etc.)
- Prompt templates
- Corpus hash
- Tool/library versions
"""

from __future__ import annotations

import hashlib
import json
import importlib.metadata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# =============================================================================
# Version Detection
# =============================================================================

def get_tool_versions() -> dict[str, str]:
    """Get versions of key libraries for reproducibility."""
    versions = {}
    
    packages = [
        "sentence-transformers",
        "transformers",
        "torch",
        "numpy",
        "faiss-cpu",
        "rank-bm25",
        "openai",
        "anthropic",
        "langchain",
        "ragas",
    ]
    
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    
    return versions


def compute_corpus_hash(document_hashes: list[str]) -> str:
    """Compute a deterministic hash of the corpus.
    
    Args:
        document_hashes: List of content hashes for each document
        
    Returns:
        SHA-256 hash of the sorted document hashes
    """
    if not document_hashes:
        return "empty_corpus"
    
    # Sort for deterministic ordering
    sorted_hashes = sorted(document_hashes)
    combined = "|".join(sorted_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# =============================================================================
# Config Snapshot
# =============================================================================

@dataclass(frozen=True)
class RetrievalSnapshot:
    """Immutable snapshot of retrieval configuration."""
    dense_k: int
    sparse_k: int
    hybrid_alpha: float
    reranker_model: Optional[str]
    reranker_top_k: int
    use_colbert: bool
    use_raptor: bool
    use_graph_rag: bool
    use_hyde: bool
    use_compression: bool
    chunk_size: int
    chunk_overlap: int
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RetrievalSnapshot":
        """Create snapshot from config dictionary."""
        return cls(
            dense_k=config.get("dense_k", 10),
            sparse_k=config.get("sparse_k", 10),
            hybrid_alpha=config.get("hybrid_alpha", 0.5),
            reranker_model=config.get("reranker_model"),
            reranker_top_k=config.get("reranker_top_k", 5),
            use_colbert=config.get("use_colbert", False),
            use_raptor=config.get("use_raptor", False),
            use_graph_rag=config.get("use_graph_rag", False),
            use_hyde=config.get("use_hyde", False),
            use_compression=config.get("use_compression", False),
            chunk_size=config.get("chunk_size", 512),
            chunk_overlap=config.get("chunk_overlap", 50),
        )


@dataclass(frozen=True)
class ModelSnapshot:
    """Immutable snapshot of model configuration."""
    provider: str
    model_id: str
    model_version: Optional[str]
    embedding_model: str
    temperature: float
    max_tokens: int
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelSnapshot":
        """Create snapshot from config dictionary."""
        return cls(
            provider=config.get("provider", "unknown"),
            model_id=config.get("model_id", "unknown"),
            model_version=config.get("model_version"),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2048),
        )


@dataclass(frozen=True)
class PromptSnapshot:
    """Immutable snapshot of prompt templates."""
    system_prompt_hash: str
    query_template_hash: str
    citation_template_hash: str
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_prompts(
        cls,
        system_prompt: str,
        query_template: str,
        citation_template: str,
    ) -> "PromptSnapshot":
        """Create snapshot from prompt strings."""
        return cls(
            system_prompt_hash=hashlib.sha256(system_prompt.encode()).hexdigest()[:12],
            query_template_hash=hashlib.sha256(query_template.encode()).hexdigest()[:12],
            citation_template_hash=hashlib.sha256(citation_template.encode()).hexdigest()[:12],
        )


@dataclass(frozen=True)
class ConfigSnapshot:
    """Immutable configuration snapshot for reproducibility.
    
    This captures the complete state of the system at query time,
    enabling exact reproduction of results.
    """
    snapshot_id: str  # SHA-256 of contents
    timestamp: str  # ISO format
    
    # Configuration components
    model: ModelSnapshot
    retrieval: RetrievalSnapshot
    prompts: PromptSnapshot
    
    # Corpus state
    corpus_hash: str
    corpus_doc_count: int
    
    # Environment
    tool_versions: tuple[tuple[str, str], ...]  # Hashable version of dict
    random_seed: Optional[int]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "model": self.model.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "prompts": self.prompts.to_dict(),
            "corpus_hash": self.corpus_hash,
            "corpus_doc_count": self.corpus_doc_count,
            "tool_versions": dict(self.tool_versions),
            "random_seed": self.random_seed,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigSnapshot":
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            timestamp=data["timestamp"],
            model=ModelSnapshot(**data["model"]),
            retrieval=RetrievalSnapshot(**data["retrieval"]),
            prompts=PromptSnapshot(**data["prompts"]),
            corpus_hash=data["corpus_hash"],
            corpus_doc_count=data["corpus_doc_count"],
            tool_versions=tuple(sorted(data["tool_versions"].items())),
            random_seed=data.get("random_seed"),
        )
    
    @classmethod
    def create(
        cls,
        model_config: dict[str, Any],
        retrieval_config: dict[str, Any],
        system_prompt: str,
        query_template: str,
        citation_template: str,
        corpus_doc_hashes: list[str],
        random_seed: Optional[int] = None,
    ) -> "ConfigSnapshot":
        """Create a new config snapshot from current configuration.
        
        Args:
            model_config: Model-related configuration
            retrieval_config: Retrieval-related configuration
            system_prompt: The system prompt template
            query_template: The query formatting template
            citation_template: The citation formatting template
            corpus_doc_hashes: List of document content hashes
            random_seed: Optional random seed for deterministic behavior
            
        Returns:
            Immutable ConfigSnapshot with computed ID
        """
        timestamp = datetime.utcnow().isoformat()
        
        model = ModelSnapshot.from_config(model_config)
        retrieval = RetrievalSnapshot.from_config(retrieval_config)
        prompts = PromptSnapshot.from_prompts(
            system_prompt, query_template, citation_template
        )
        
        corpus_hash = compute_corpus_hash(corpus_doc_hashes)
        tool_versions = tuple(sorted(get_tool_versions().items()))
        
        # Compute snapshot ID from all components
        id_content = json.dumps({
            "model": model.to_dict(),
            "retrieval": retrieval.to_dict(),
            "prompts": prompts.to_dict(),
            "corpus_hash": corpus_hash,
            "tool_versions": dict(tool_versions),
        }, sort_keys=True)
        snapshot_id = hashlib.sha256(id_content.encode()).hexdigest()[:16]
        
        return cls(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            model=model,
            retrieval=retrieval,
            prompts=prompts,
            corpus_hash=corpus_hash,
            corpus_doc_count=len(corpus_doc_hashes),
            tool_versions=tool_versions,
            random_seed=random_seed,
        )


# =============================================================================
# Snapshot Store
# =============================================================================

class ConfigSnapshotStore:
    """Persistent storage for configuration snapshots."""
    
    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or Path("data/config_snapshots.json")
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._load()
    
    def _load(self) -> None:
        """Load snapshots from disk."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    self._snapshots = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load snapshots: {e}")
                self._snapshots = {}
    
    def _save(self) -> None:
        """Save snapshots to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._snapshots, f, indent=2)
    
    def save(self, snapshot: ConfigSnapshot) -> str:
        """Save a snapshot and return its ID."""
        self._snapshots[snapshot.snapshot_id] = snapshot.to_dict()
        self._save()
        return snapshot.snapshot_id
    
    def get(self, snapshot_id: str) -> Optional[ConfigSnapshot]:
        """Get a snapshot by ID."""
        data = self._snapshots.get(snapshot_id)
        if data:
            return ConfigSnapshot.from_dict(data)
        return None
    
    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent snapshots with summary info."""
        # Sort by timestamp descending
        sorted_snapshots = sorted(
            self._snapshots.values(),
            key=lambda x: x["timestamp"],
            reverse=True,
        )
        
        return [
            {
                "snapshot_id": s["snapshot_id"],
                "timestamp": s["timestamp"],
                "corpus_hash": s["corpus_hash"],
                "model_id": s["model"]["model_id"],
            }
            for s in sorted_snapshots[:limit]
        ]
    
    def delete_old(self, keep_count: int = 100) -> int:
        """Delete old snapshots, keeping the most recent ones."""
        if len(self._snapshots) <= keep_count:
            return 0
        
        sorted_ids = sorted(
            self._snapshots.keys(),
            key=lambda x: self._snapshots[x]["timestamp"],
            reverse=True,
        )
        
        to_delete = sorted_ids[keep_count:]
        for snapshot_id in to_delete:
            del self._snapshots[snapshot_id]
        
        self._save()
        return len(to_delete)


# =============================================================================
# Singleton
# =============================================================================

_snapshot_store: Optional[ConfigSnapshotStore] = None


def get_snapshot_store() -> ConfigSnapshotStore:
    """Get the global snapshot store instance."""
    global _snapshot_store
    if _snapshot_store is None:
        _snapshot_store = ConfigSnapshotStore()
    return _snapshot_store


__all__ = [
    "ConfigSnapshot",
    "ModelSnapshot",
    "RetrievalSnapshot",
    "PromptSnapshot",
    "ConfigSnapshotStore",
    "get_snapshot_store",
    "get_tool_versions",
    "compute_corpus_hash",
]
