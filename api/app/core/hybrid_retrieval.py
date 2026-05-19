"""Hybrid retrieval engine with dense embeddings, BM25, and reranking.

This module provides a state-of-the-art retrieval implementation that combines:
- Dense vector search using sentence-transformers
- BM25 sparse retrieval for keyword matching
- Reciprocal Rank Fusion (RRF) for combining results
- Cross-encoder reranking for precision
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from .cache import get_cache_manager
from .chunking import Chunk, ChunkingStrategy, get_chunker
from .documents import Document, DocumentStore
from .hierarchy import DocumentTree
from .secrets_vault import get_secrets_vault

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoTokenizer = AutoModel = None

# Lazy imports for optional dependencies
_sentence_transformer = None
_cross_encoder = None
_bm25_class = None


def _get_sentence_transformer():
    """Lazy load sentence-transformers."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer
        except ImportError:
            _sentence_transformer = False
    return _sentence_transformer if _sentence_transformer else None


def _get_cross_encoder():
    """Lazy load cross-encoder."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder
        except ImportError:
            _cross_encoder = False
    return _cross_encoder if _cross_encoder else None


def _get_bm25():
    """Lazy load BM25."""
    global _bm25_class
    if _bm25_class is None:
        try:
            from rank_bm25 import BM25Okapi
            _bm25_class = BM25Okapi
        except ImportError:
            _bm25_class = False
    return _bm25_class if _bm25_class else None


class RemoteEmbeddingClient:
    """HTTP-based embedding client for API-backed models."""

    supports_semantic_chunking = False

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def encode(self, texts, convert_to_numpy: bool = False, show_progress_bar: bool = False):
        inputs = [texts] if isinstance(texts, str) else list(texts)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": inputs}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding", []) for item in items]

        if convert_to_numpy:
            return np.array(embeddings, dtype=np.float32)
        return embeddings


@dataclass
class RetrievalResult:
    """A single retrieval result with score and source info."""
    document: Document
    score: float
    chunk_text: str
    retrieval_method: str = "hybrid"
    # Span-level citation metadata
    chunk_id: str = ""       # Unique ID for this chunk (doc_id-chunk_index)
    start_char: int = 0      # Start offset in original document text
    end_char: int = 0        # End offset in original document text


class EmbeddingModelPreset:
    """2025 MTEB-validated embedding model presets.

    These models are selected based on MTEB leaderboard performance
    as of late 2025, balancing accuracy, speed, and resource usage.
    """
    # Solid baseline - good balance of speed and accuracy
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    # Multi-lingual, multi-granularity - excellent for diverse corpora
    BGE_M3 = "BAAI/bge-m3"
    # MTEB leader - instruction-tuned for high accuracy (larger model)
    GTE_QWEN = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    # High accuracy with long context support
    E5_MISTRAL = "intfloat/e5-mistral-7b-instruct"
    # OpenAI API model (requires OPENAI_API_KEY)
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    # Fast/lightweight for resource-constrained environments
    SENTENCE_MINI = "sentence-transformers/all-MiniLM-L6-v2"

    # Model metadata for auto-configuration
    MODEL_INFO = {
        BGE_BASE: {"dimensions": 768, "max_tokens": 512, "requires_api": False},
        BGE_M3: {"dimensions": 1024, "max_tokens": 8192, "requires_api": False},
        GTE_QWEN: {"dimensions": 1536, "max_tokens": 32768, "requires_api": False},
        E5_MISTRAL: {"dimensions": 4096, "max_tokens": 32768, "requires_api": False},
        TEXT_EMBEDDING_3_LARGE: {"dimensions": 3072, "max_tokens": 8191, "requires_api": True},
        SENTENCE_MINI: {"dimensions": 384, "max_tokens": 512, "requires_api": False},
    }

    @classmethod
    def get_info(cls, model: str) -> dict:
        """Get model metadata for configuration."""
        return cls.MODEL_INFO.get(model, {"dimensions": 768, "max_tokens": 512, "requires_api": False})


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""
    # Model configuration - use preset for easy 2025 model selection
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    model_preset: str | None = None  # Set to EmbeddingModelPreset value to auto-configure
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval parameters
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    use_reranking: bool = True
    rerank_top_k: int = 20  # Candidates for reranking
    recency_weight: float = 0.1
    recency_half_life_days: float = 90.0
    title_boost: float = 0.6
    heading_boost: float = 0.4
    proximity_weight: float = 0.5
    diversity: float = 0.0

    # Chunking
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    chunk_size: int = 400
    chunk_overlap: int = 50

    # Feature toggles
    raptor: bool = False
    graph: bool = False
    use_colbert: bool = False
    colbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    colbert_top_k: int = 12
    deployment_profile: str = "local_only"
    backend_map: dict[str, str] = field(default_factory=dict)
    fallback_map: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        """Apply model preset if specified."""
        if self.model_preset:
            self.embedding_model = self.model_preset


class HybridRetrievalEngine:
    """Advanced retrieval engine with hybrid search and reranking.

    Combines dense vector search (semantic) with BM25 (keyword) retrieval,
    then optionally applies cross-encoder or ColBERT-style reranking for maximum precision.

    Supports disk persistence for indexes to avoid rebuild on restart.
    """

    def __init__(
        self,
        documents: DocumentStore,
        config: HybridConfig | None = None,
        persist_path: str | None = "data/indexes",
    ) -> None:
        self._docs = documents
        self._config = config or HybridConfig()
        self._persist_path = persist_path

        # Models (lazy loaded)
        self._embedder: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None
        self._embedder_failed = False
        self._reranker_failed = False

        # Index data
        self._chunks: list[tuple[str, Chunk]] = []  # (doc_id, chunk)
        self._embeddings: np.ndarray | None = None
        self._bm25 = None
        self._tokenized_corpus: list[list[str]] = []
        self._last_cache: dict[str, Any] = {}
        self._cache_manager = get_cache_manager()
        # Precision modes
        self._colbert_tokenizer = None
        self._colbert_model = None
        self._colbert_failed = False
        self._colbert_device = "cpu"
        self._last_colbert_stats: dict[str, Any] = {"applied": False}
        # Metadata-aware scoring
        self._doc_title_tokens: dict[str, set[str]] = {}
        self._chunk_heading_tokens: dict[int, set[str]] = {}
        self._doc_timestamps: dict[str, datetime] = {}

        # Persistence and corpus versioning (G3: Cache never stale)
        self._index_persistence = None
        self._corpus_version: str = ""
        self._config_hash: str = ""
        self._corpus_version_counter: int = 0  # Monotonically increasing on any change
        self._doc_content_hashes: dict[str, str] = {}  # doc_id -> content_hash for incremental
        self._index_lock = threading.RLock()

        # Phase 4/6/5: Hierarchical & Graph structures
        self._trees: dict[str, Any] = {}  # doc_id -> DocumentTree
        self._graph: dict[str, set[int]] = {}  # term -> set of chunk indices
        self._graph_rag: Any | None = None
        self._graph_ready: bool = False
        self._graph_failed: bool = False

        # Load models
        self._init_models()

    def _init_models(self) -> None:
        """Lazy load heavy models."""
        if self._config.embedding_model and not self._embedder_failed and self._embedder is None:
            model_info = EmbeddingModelPreset.get_info(self._config.embedding_model)
            if model_info.get("requires_api"):
                vault = get_secrets_vault()
                api_key = vault.get("OPENAI_API_KEY") or ""
                base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                if not api_key:
                    print("Error: OPENAI_API_KEY not set for API-backed embedding model.")
                    self._embedder_failed = True
                else:
                    self._embedder = RemoteEmbeddingClient(
                        model=self._config.embedding_model,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    print(f"Embedding model configured via API: {self._config.embedding_model} ({base_url})")
            else:
                SentenceTransformer = _get_sentence_transformer()
                if SentenceTransformer:
                    try:
                        is_local = os.path.isdir(self._config.embedding_model)
                        location = "Local path" if is_local else "HuggingFace cache/remote"
                        print(f"Loading embedding model: {self._config.embedding_model} ({location})...")
                        try:
                            # Force non-meta device loading for CPU stability
                            self._embedder = SentenceTransformer(
                                self._config.embedding_model,
                                device="cpu",
                                model_kwargs={"low_cpu_mem_usage": False, "device_map": None}
                            )
                        except (TypeError, Exception):
                            # Fallback for older versions or unexpected errors
                            self._embedder = SentenceTransformer(
                                self._config.embedding_model,
                                device="cpu"
                            )
                        print(f"Embedding model loaded successfully from {location}.")
                    except Exception as e:
                        print(f"Error: Could not load embedding model: {e}")
                        self._embedder = None
                        self._embedder_failed = True
                else:
                    print("Warning: sentence-transformers not installed. Dense retrieval disabled.")

        if self._config.use_reranking and self._config.reranker_model and not self._reranker_failed and self._reranker is None:
            CrossEncoder = _get_cross_encoder()
            if CrossEncoder:
                try:
                    is_local = os.path.isdir(self._config.reranker_model)
                    location = "Local path" if is_local else "HuggingFace cache/remote"
                    print(f"Loading reranker model: {self._config.reranker_model} ({location})...")
                    try:
                        # Force non-meta device loading for CPU stability
                        self._reranker = CrossEncoder(
                            self._config.reranker_model,
                            device="cpu",
                            # Note: CrossEncoder usually kwargs are passed to AutoModel,
                            # checking support varies but this is the safest 'meta' fix attempt
                            model_kwargs={"low_cpu_mem_usage": False, "device_map": None}
                        )
                    except (TypeError, Exception):
                        self._reranker = CrossEncoder(
                            self._config.reranker_model,
                            device="cpu"
                        )
                    print(f"Reranker model loaded successfully from {location}.")
                except Exception as e:
                    print(f"Warning: Could not load reranker model: {e}")
                    self._reranker = None
                    self._reranker_failed = True

    def _requires_rebuild(self, config: HybridConfig) -> bool:
        """Check if config changes require index rebuild."""
        fields = ("embedding_model", "chunking_strategy", "chunk_size", "chunk_overlap")
        return any(getattr(self._config, f) != getattr(config, f) for f in fields)

    def reconfigure(self, config: HybridConfig, rebuild: bool | None = None) -> bool:
        """Apply a new retrieval config and rebuild indexes if needed.

        Returns True if a rebuild was triggered.
        """
        with self._index_lock:
            previous = self._config
            rebuild_needed = self._requires_rebuild(config)
            graph_enabled = config.graph and not previous.graph
            raptor_enabled = config.raptor and not previous.raptor

            self._config = config

            if previous.embedding_model != config.embedding_model:
                self._embedder = None
                self._embedder_failed = False
            if (
                previous.reranker_model != config.reranker_model
                or previous.use_reranking != config.use_reranking
            ):
                self._reranker = None
                self._reranker_failed = False
            if previous.use_colbert != config.use_colbert or previous.colbert_model != config.colbert_model:
                self._colbert_model = None
                self._colbert_tokenizer = None
                self._colbert_failed = False

        self._init_models()

        if rebuild is True or (rebuild is None and rebuild_needed):
            if not self.load_index():
                self.build()
            return True

        if graph_enabled:
            self.build()
            return True

        if raptor_enabled and self._chunks:
            self._rebuild_trees_parallel()

        return False

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()

    def _tokenize_terms(self, text: str) -> list[str]:
        """Tokenize text into alphanumeric terms for scoring boosts."""
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t]

    def _parse_timestamp(self, value: str) -> datetime | None:
        """Parse common timestamp formats into timezone-aware UTC datetime."""
        if not value:
            return None
        raw = value.strip()
        if not raw:
            return None
        # Epoch seconds or milliseconds
        if re.fullmatch(r"\d{10,}", raw):
            try:
                epoch = int(raw)
                if len(raw) > 10:
                    epoch = int(epoch / 1000)
                return datetime.fromtimestamp(epoch, tz=UTC)
            except Exception:
                return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except Exception:
            return None

    def _extract_doc_timestamp(self, metadata: dict[str, str]) -> datetime | None:
        """Extract the most recent timestamp from document metadata."""
        if not metadata:
            return None
        keys = (
            "updated_at",
            "published_at",
            "created_at",
            "uploaded_at",
            "processed_at",
            "timestamp",
            "date",
        )
        best: datetime | None = None
        for key in keys:
            value = metadata.get(key)
            if not value:
                continue
            parsed = self._parse_timestamp(value)
            if parsed and (best is None or parsed > best):
                best = parsed
        return best

    def _build_field_tokens(self, docs: list[Document]) -> None:
        """Precompute title and heading tokens for field-aware scoring."""
        self._doc_title_tokens = {
            doc.id: set(self._tokenize_terms(doc.title))
            for doc in docs
            if doc.title
        }
        self._chunk_heading_tokens = {}
        if not self._trees:
            return
        # Map (doc_id, chunk_index) -> global index in self._chunks
        chunk_index_map: dict[tuple[str, int], int] = {}
        for idx, (doc_id, chunk) in enumerate(self._chunks):
            chunk_index_map[(doc_id, chunk.index)] = idx
        for doc_id, tree in self._trees.items():
            for node in tree.nodes.values():
                if not node.chunk_ids:
                    continue
                headings = [node.title]
                for ancestor in tree.get_ancestors(node.id):
                    headings.append(ancestor.title)
                heading_tokens: set[str] = set()
                for heading in headings:
                    heading_tokens.update(self._tokenize_terms(heading))
                if not heading_tokens:
                    continue
                for chunk_id in node.chunk_ids:
                    try:
                        chunk_index = int(chunk_id)
                    except ValueError:
                        continue
                    global_idx = chunk_index_map.get((doc_id, chunk_index))
                    if global_idx is None:
                        continue
                    existing = self._chunk_heading_tokens.get(global_idx, set())
                    self._chunk_heading_tokens[global_idx] = existing | heading_tokens

    def _normalize_routing_params(self, routing_params: dict[str, Any] | None) -> dict[str, float]:
        """Normalize per-query routing params and merge with defaults."""
        params = dict(routing_params or {})
        sparse_weight = float(params.get("sparse_weight", self._config.sparse_weight) or 0.0)
        dense_weight = float(params.get("dense_weight", self._config.dense_weight) or 0.0)
        if "dense_weight" not in params and "sparse_weight" in params:
            dense_weight = max(0.0, 1.0 - sparse_weight)
        total = dense_weight + sparse_weight
        if total > 0:
            dense_weight /= total
            sparse_weight /= total
        literal_weight = float(params.get("literal_weight", max(0.2, min(0.6, sparse_weight))))
        return {
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "literal_weight": literal_weight,
            "diversity": float(params.get("diversity", self._config.diversity) or 0.0),
            "title_boost": float(params.get("title_boost", self._config.title_boost) or 0.0),
            "heading_boost": float(params.get("heading_boost", self._config.heading_boost) or 0.0),
            "proximity_weight": float(params.get("proximity_weight", self._config.proximity_weight) or 0.0),
            "recency_weight": float(params.get("recency_weight", self._config.recency_weight) or 0.0),
            "recency_half_life_days": float(
                params.get("recency_half_life_days", self._config.recency_half_life_days) or 1.0
            ),
        }

    def _field_boost(
        self,
        idx: int,
        query_terms: set[str],
        title_boost: float,
        heading_boost: float,
    ) -> float:
        """Compute field-aware boost for title and heading matches."""
        if not query_terms:
            return 0.0
        doc_id = self._chunks[idx][0] if idx < len(self._chunks) else ""
        title_terms = self._doc_title_tokens.get(doc_id, set())
        heading_terms = self._chunk_heading_tokens.get(idx, set())
        title_hits = len(query_terms & title_terms) if title_terms else 0
        heading_hits = len(query_terms & heading_terms) if heading_terms else 0
        return (title_hits * title_boost) + (heading_hits * heading_boost)

    def _term_proximity(
        self,
        query_terms: set[str],
        chunk_terms: list[str],
    ) -> float:
        """Compute a proximity score based on minimal term window."""
        if len(query_terms) < 2 or not chunk_terms:
            return 0.0
        positions: list[tuple[int, str]] = []
        for idx, term in enumerate(chunk_terms):
            if term in query_terms:
                positions.append((idx, term))
        if len(positions) < 2:
            return 0.0
        positions.sort()
        needed = set(query_terms)
        counts: dict[str, int] = {}
        covered = 0
        left = 0
        best_window: int | None = None
        for right, (_, term) in enumerate(positions):
            if counts.get(term, 0) == 0:
                covered += 1
            counts[term] = counts.get(term, 0) + 1
            while covered == len(needed) and left <= right:
                window_size = positions[right][0] - positions[left][0] + 1
                if best_window is None or window_size < best_window:
                    best_window = window_size
                left_term = positions[left][1]
                counts[left_term] -= 1
                if counts[left_term] == 0:
                    covered -= 1
                left += 1
        if best_window is None:
            return 0.0
        return 1.0 / (1.0 + float(best_window))

    def _apply_recency_boost(
        self,
        results: list[tuple[int, float]],
        recency_weight: float,
        half_life_days: float,
    ) -> list[tuple[int, float]]:
        """Apply recency prior to result scores."""
        if recency_weight <= 0 or half_life_days <= 0:
            return results
        now = datetime.now(UTC)
        adjusted: list[tuple[int, float]] = []
        with self._index_lock:
            for idx, score in results:
                doc_id = self._chunks[idx][0] if idx < len(self._chunks) else ""
                ts = self._doc_timestamps.get(doc_id)
                if not ts:
                    adjusted.append((idx, score))
                    continue
                age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
                decay = math.exp(-math.log(2) * age_days / half_life_days)
                adjusted.append((idx, score * (1.0 + (recency_weight * decay))))
        return adjusted

    def _apply_diversity_rerank(
        self,
        candidates: list[tuple[int, float]],
        top_k: int,
        diversity: float,
    ) -> list[tuple[int, float]]:
        """Apply a lightweight diversity rerank using token overlap."""
        if diversity <= 0 or len(candidates) <= 1:
            return candidates[:top_k]
        pool = candidates[: max(top_k * 4, top_k)]
        selected: list[tuple[int, float]] = []
        token_cache: dict[int, set[str]] = {}

        def tokens_for(idx: int) -> set[str]:
            if idx in token_cache:
                return token_cache[idx]
            if idx >= len(self._chunks):
                token_cache[idx] = set()
                return token_cache[idx]
            _, chunk = self._chunks[idx]
            token_cache[idx] = set(self._tokenize_terms(chunk.text))
            return token_cache[idx]

        with self._index_lock:
            while pool and len(selected) < top_k:
                best = None
                best_score = None
                for idx, score in pool:
                    candidate_tokens = tokens_for(idx)
                    max_sim = 0.0
                    if selected and candidate_tokens:
                        for sel_idx, _ in selected:
                            sel_tokens = tokens_for(sel_idx)
                            if not sel_tokens:
                                continue
                            overlap = len(candidate_tokens & sel_tokens)
                            union = len(candidate_tokens | sel_tokens)
                            if union:
                                max_sim = max(max_sim, overlap / union)
                    mmr_score = ((1 - diversity) * score) - (diversity * max_sim)
                    if best_score is None or mmr_score > best_score:
                        best_score = mmr_score
                        best = (idx, score)
                if best is None:
                    break
                selected.append(best)
                pool = [c for c in pool if c[0] != best[0]]
        return selected

    def model_status(self) -> dict[str, bool]:
        """Return availability of dense and rerank models."""
        return {
            "dense_enabled": self._embedder is not None,
            "reranker_enabled": self._reranker is not None and self._config.use_reranking,
        }

    # =========================================================================
    # Corpus Version Management (Guarantee G3: Cache never stale)
    # =========================================================================

    def get_corpus_version(self) -> str:
        """Return current corpus version as string.

        Used in cache keys to ensure stale results are never returned
        after corpus changes.
        """
        return str(self._corpus_version_counter)

    def increment_corpus_version(self) -> int:
        """Increment corpus version counter on any corpus change.

        Called on document ingest, delete, or re-chunk operations.
        Returns the new version number.
        """
        self._corpus_version_counter += 1
        # Also invalidate hash-based version since corpus changed
        self._corpus_version = ""
        return self._corpus_version_counter

    def get_retrieval_mode_flags(self) -> int:
        """Get current retrieval mode as bitmask for cache keys.

        Returns an integer bitmask indicating which retrieval modes are enabled.
        Used in cache keys to prevent stale hits when modes are toggled.
        """
        from .cache import RetrievalMode
        return int(RetrievalMode.from_config(
            raptor=self._config.raptor,
            graph=self._config.graph,
            rerank=self._config.use_reranking,
            colbert=self._config.use_colbert,
        ))

    def compute_content_hash(self, text: str) -> str:
        """Compute SHA-256 hash of document content for change detection.

        Used by incremental ingestion to skip re-processing unchanged docs.
        """
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def is_doc_changed(self, doc_id: str, content_hash: str) -> bool:
        """Check if document content has changed since last indexing.

        Args:
            doc_id: The document identifier
            content_hash: Hash of current document content

        Returns:
            True if document is new or changed, False if unchanged
        """
        existing_hash = self._doc_content_hashes.get(doc_id)
        return existing_hash != content_hash

    def register_doc_hash(self, doc_id: str, content_hash: str) -> None:
        """Register document content hash after successful indexing."""
        self._doc_content_hashes[doc_id] = content_hash

    def clear_cache(self, include_disk: bool = True) -> bool:
        """Clear all cached indexes and reset to fresh state.

        Args:
            include_disk: If True, also delete persisted index files on disk.

        Returns:
            True if cache was cleared successfully.
        """
        # Delete persisted indexes if requested
        if include_disk:
            persistence = self._get_persistence()
            if persistence:
                try:
                    persistence.delete_index("default")
                    print("HybridRetrievalEngine: Cleared disk cache")
                except Exception as e:
                    print(f"HybridRetrievalEngine: Failed to clear disk cache: {e}")

        # Clear in-memory indexes
        with self._index_lock:
            self._chunks = []
            self._embeddings = None
            self._bm25 = None
            self._tokenized_corpus = []
            self._trees = {}
            self._graph = {}
            self._doc_content_hashes = {}

        # Increment corpus version to invalidate query caches
        self.increment_corpus_version()

        print("HybridRetrievalEngine: Cache cleared")
        return True

    def build(
        self,
        on_progress: Callable[[str, int, int, str | None], None] | None = None,
    ) -> None:
        """Build the retrieval index from all documents.

        Args:
            on_progress: Optional callback (stage, current, total) for progress updates.
                        stage: Name of current processing stage
                        current: Current item being processed (0-indexed)
                        total: Total items to process
        """
        docs = self._docs.list()
        print(f"HybridRetrievalEngine: Building index for {len(docs)} documents...")

        def emit(stage: str, current: int, total: int) -> None:
            if on_progress:
                on_progress(stage, current, total)

        emit("loading_documents", 0, len(docs))

        if not docs:
            print("HybridRetrievalEngine: No documents found in store.")
            self._chunks = []
            self._embeddings = None
            self._bm25 = None
            return

        # Get chunker
        chunker = get_chunker(
            strategy=self._config.chunking_strategy,
            embedder=self._embedder,
            target_size=self._config.chunk_size,
            overlap=self._config.chunk_overlap,
        )

        # Build chunks from all documents
        self._chunks = []
        corpus_texts: list[str] = []

        for idx, doc in enumerate(docs):
            emit("chunking_documents", idx, len(docs))
            if not doc.text.strip():
                print(f"HybridRetrievalEngine: Skipping empty document: {doc.title}")
                continue

            chunks = chunker.chunk(doc.text)
            print(f"HybridRetrievalEngine: Processing '{doc.title}' -> {len(chunks)} chunks")
            for chunk in chunks:
                self._chunks.append((doc.id, chunk))
                corpus_texts.append(chunk.text)

        emit("chunking_documents", len(docs), len(docs))

        if not corpus_texts:
            print("HybridRetrievalEngine: No text chunks generated (all docs might be empty).")
            return

        # Build dense embeddings with batching for progress feedback
        if self._embedder:
            try:
                batch_size = 64
                all_embeddings = []
                total_chunks = len(corpus_texts)

                for i in range(0, total_chunks, batch_size):
                    batch = corpus_texts[i:i + batch_size]
                    batch_embeddings = self._embedder.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                    all_embeddings.append(batch_embeddings)
                    emit("embedding_chunks", min(i + batch_size, total_chunks), total_chunks)

                if all_embeddings:
                    self._embeddings = np.vstack(all_embeddings)
                else:
                    self._embeddings = None
            except Exception as e:
                print(f"Warning: Embedding failed: {e}")
                self._embeddings = None

        emit("building_sparse_index", 0, 1)

        # Tokenize corpus for sparse retrieval (BM25 or fallback overlap scoring)
        self._tokenized_corpus = [self._tokenize(text) for text in corpus_texts]

        # Build BM25 index if available
        BM25Class = _get_bm25()
        if BM25Class:
            try:
                self._bm25 = BM25Class(self._tokenized_corpus)
            except Exception as e:
                print(f"Warning: BM25 indexing failed: {e}")
                self._bm25 = None
        elif not self._tokenized_corpus:
            self._bm25 = None

        emit("building_sparse_index", 1, 1)

        # Build hierarchy in parallel if enabled
        if self._config.raptor or self._config.chunking_strategy != ChunkingStrategy.FIXED:
            self._rebuild_trees_parallel(on_progress=on_progress)

        # Build GraphRAG if enabled (restricted to max chunks for build time)
        if self._config.graph:
            emit("building_knowledge_graph", 0, 1)
            try:
                # Use a background loop for async GraphRAG building if we're in a sync context
                from .gatherer import EvidenceChunk
                from .graph_rag import GraphRAG
                from .providers import ProviderFactory

                # We need a provider for building. Default to OpenAI if not fixed.
                # In JR-AutoRAG, Orchestrator usually provides this, but here we try factory.
                factory = ProviderFactory()
                provider = factory.get_default_provider() # Simplified for build phase

                if provider:
                    self._graph_rag = GraphRAG()
                    max_graph_chunks = 100 # Safety limit for build phase
                    evidence = [
                        EvidenceChunk(
                            id=f"{did}-{c.index}",
                            title=did,
                            snippet=c.text,
                            score=1.0
                        )
                        for did, c in self._chunks[:max_graph_chunks]
                    ]

                    async def build_g():
                        await self._graph_rag.build_from_chunks(evidence, provider)
                        self._graph_rag.detect_communities()
                        await self._graph_rag.summarize_communities(provider)
                        self._graph_ready = True

                    try:
                        try:
                            asyncio.get_running_loop()
                        except RuntimeError:
                            asyncio.run(build_g())
                        else:
                            error: Exception | None = None
                            done = threading.Event()

                            def runner():
                                nonlocal error
                                try:
                                    asyncio.run(build_g())
                                except Exception as exc:
                                    error = exc
                                finally:
                                    done.set()

                            thread = threading.Thread(
                                target=runner,
                                name="autorag-graph-build",
                                daemon=True,
                            )
                            thread.start()
                            done.wait()
                            if error:
                                raise error
                    except Exception as e:
                        print(f"Warning: GraphRAG building async execution failed: {e}")
                        self._graph_failed = True
            except Exception as e:
                print(f"Warning: GraphRAG indexing failed: {e}")
                self._graph_failed = True
            emit("building_knowledge_graph", 1, 1)

        emit("building_keyword_graph", 0, 1)

        # Build simple keyword graph for context expansion
        if self._tokenized_corpus:
            self._graph = {}
            for i, tokens in enumerate(self._tokenized_corpus):
                # Focus on significant terms (nouns/entities - simplified)
                for token in set(tokens):
                    if len(token) > 4: # Heuristic for keywords
                        if token not in self._graph:
                            self._graph[token] = set()
                        self._graph[token].add(i)

        emit("building_keyword_graph", 1, 1)
        emit("building_metadata_cache", 0, 1)

        # Metadata-aware scoring caches
        self._doc_timestamps = {}
        for doc in docs:
            ts = self._extract_doc_timestamp(doc.metadata or {})
            if ts:
                self._doc_timestamps[doc.id] = ts
        self._build_field_tokens(docs)

        emit("building_metadata_cache", 1, 1)

        # Auto-save if persistence enabled
        if self._persist_path and self._chunks:
            emit("saving_index", 0, 1)
            self.save_index()
            emit("saving_index", 1, 1)

    def index_documents(self, docs: list[Document]) -> None:
        """Incrementally index new/updated documents without full rebuild."""
        if not docs:
            return

        doc_ids = {doc.id for doc in docs if doc.id}
        with self._index_lock:
            if doc_ids and self._chunks:
                keep_indices = [i for i, (doc_id, _) in enumerate(self._chunks) if doc_id not in doc_ids]
                if len(keep_indices) != len(self._chunks):
                    self._chunks = [self._chunks[i] for i in keep_indices]
                    if self._embeddings is not None:
                        try:
                            self._embeddings = self._embeddings[keep_indices]
                        except Exception:
                            self._embeddings = None
                    if self._tokenized_corpus:
                        self._tokenized_corpus = [self._tokenized_corpus[i] for i in keep_indices]
                    BM25Class = _get_bm25()
                    if BM25Class and self._tokenized_corpus:
                        try:
                            self._bm25 = BM25Class(self._tokenized_corpus)
                        except Exception as e:
                            print(f"Warning: BM25 rebuild failed after doc removal: {e}")
                            self._bm25 = None
                    else:
                        self._bm25 = None
                    self._graph = {}
                    for i, tokens in enumerate(self._tokenized_corpus):
                        for token in set(tokens):
                            if len(token) > 4:
                                self._graph.setdefault(token, set()).add(i)
                    for doc_id in doc_ids:
                        self._trees.pop(doc_id, None)
                        self._doc_timestamps.pop(doc_id, None)
                        self._doc_content_hashes.pop(doc_id, None)
                    self._chunk_heading_tokens = {}

        self._init_models()
        chunker = get_chunker(
            strategy=self._config.chunking_strategy,
            embedder=self._embedder,
            target_size=self._config.chunk_size,
            overlap=self._config.chunk_overlap,
        )

        new_chunks: list[tuple[str, Chunk]] = []
        new_texts: list[str] = []

        for doc in docs:
            if not doc.text.strip():
                continue
            chunks = chunker.chunk(doc.text)
            for chunk in chunks:
                new_chunks.append((doc.id, chunk))
                new_texts.append(chunk.text)

        if not new_texts:
            return

        # Dense embeddings (append-only)
        if self._embedder:
            try:
                batch_embeddings = self._embedder.encode(
                    new_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                with self._index_lock:
                    if self._embeddings is None or len(self._embeddings) == 0:
                        self._embeddings = np.array(batch_embeddings)
                    else:
                        self._embeddings = np.vstack([self._embeddings, batch_embeddings])
            except Exception as e:
                print(f"Warning: Incremental embedding failed: {e}")
                with self._index_lock:
                    self._embeddings = None
        else:
            with self._index_lock:
                self._embeddings = None

        # Sparse index update
        new_tokenized = [self._tokenize(text) for text in new_texts]
        with self._index_lock:
            start_index = len(self._tokenized_corpus)
            self._tokenized_corpus.extend(new_tokenized)

        BM25Class = _get_bm25()
        if BM25Class:
            try:
                bm25 = BM25Class(self._tokenized_corpus)
            except Exception as e:
                print(f"Warning: BM25 incremental rebuild failed: {e}")
                bm25 = None
        else:
            bm25 = None
        with self._index_lock:
            self._bm25 = bm25

        # Update keyword graph for context expansion
        with self._index_lock:
            if self._tokenized_corpus:
                if not self._graph:
                    self._graph = {}
                for offset, tokens in enumerate(new_tokenized):
                    idx = start_index + offset
                    for token in set(tokens):
                        if len(token) > 4:
                            self._graph.setdefault(token, set()).add(idx)

        # Update hierarchical trees if enabled
        if self._config.raptor or self._config.chunking_strategy != ChunkingStrategy.FIXED:
            from .hierarchy import HierarchyBuilder

            hb = HierarchyBuilder()
            for doc in docs:
                if not doc.text.strip():
                    continue
                tree = hb.build(doc.text, doc.id, doc.title)
                doc_chunks = [c for did, c in new_chunks if did == doc.id]
                hb.associate_chunks(tree, doc_chunks, doc.text)
                with self._index_lock:
                    self._trees[doc.id] = tree

        # Mark GraphRAG as stale if enabled
        if self._config.graph:
            with self._index_lock:
                self._graph_ready = False
                self._graph_rag = None
                self._graph_failed = False

        # Update metadata caches
        for doc in docs:
            ts = self._extract_doc_timestamp(doc.metadata or {})
            if ts:
                with self._index_lock:
                    self._doc_timestamps[doc.id] = ts
            content_hash = (doc.metadata or {}).get("content_hash")
            if content_hash:
                with self._index_lock:
                    self.register_doc_hash(doc.id, content_hash)

        with self._index_lock:
            self._chunks.extend(new_chunks)
            self._build_field_tokens(self._docs.list())
        self.increment_corpus_version()

        if self._persist_path and self._chunks:
            self.save_index()

    def _get_persistence(self):
        """Lazy-load persistence manager."""
        if self._index_persistence is None and self._persist_path:
            from .persistence import IndexPersistence
            with self._index_lock:
                if self._index_persistence is None:
                    self._index_persistence = IndexPersistence(self._persist_path)
        return self._index_persistence

    def _compute_versions(self) -> tuple[str, str]:
        """Compute corpus version and config hash for cache invalidation."""
        import hashlib
        import json

        # Corpus version from document IDs
        doc_ids = sorted(d.id for d in self._docs.list())
        corpus_str = ",".join(doc_ids)
        corpus_version = hashlib.sha256(corpus_str.encode()).hexdigest()[:16]

        # Config hash from retrieval config
        config_dict = {
            "embedding_model": self._config.embedding_model,
            "chunk_size": self._config.chunk_size,
            "chunk_overlap": self._config.chunk_overlap,
            "chunking_strategy": str(self._config.chunking_strategy),
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        return corpus_version, config_hash

    def save_index(self, index_name: str = "default") -> bool:
        """Save current index to disk for fast reload."""
        persistence = self._get_persistence()
        if not persistence:
            return False

        with self._index_lock:
            chunks = list(self._chunks)
            embeddings = self._embeddings
            bm25 = self._bm25
            tokenized_corpus = list(self._tokenized_corpus)
            trees = {id: t.to_dict() for id, t in self._trees.items()} if self._trees else {}
            graph_data = self._graph_rag.to_dict() if self._graph_rag and self._graph_ready else None

        if not chunks or embeddings is None:
            print("HybridRetrievalEngine: No index to save")
            return False

        import time

        from .persistence import IndexMetadata

        corpus_version, config_hash = self._compute_versions()
        self._corpus_version = corpus_version
        self._config_hash = config_hash

        metadata = IndexMetadata(
            corpus_version=corpus_version,
            config_hash=config_hash,
            chunk_count=len(self._chunks),
            created_at=time.time(),
            model_name=self._config.embedding_model,
        )

        # 1. Save dense index
        try:
            persistence.save_dense_index(
                index_name=index_name,
                embeddings=embeddings,
                chunks=chunks,
                metadata=metadata,
            )
            print(f"HybridRetrievalEngine: Saved dense index ({len(chunks)} chunks)")
        except Exception as e:
            print(f"HybridRetrievalEngine: Failed to save dense index: {e}")
            return False

        # 2. Save sparse index
        if bm25 and tokenized_corpus:
            try:
                persistence.save_sparse_index(
                    index_name=index_name,
                    bm25=bm25,
                    tokenized_corpus=tokenized_corpus,
                    metadata=metadata,
                )
                print("HybridRetrievalEngine: Saved sparse index")
            except Exception as e:
                print(f"HybridRetrievalEngine: Failed to save sparse index: {e}")

        # 3. Save GraphRAG index
        if graph_data:
            try:
                persistence.save_graph(
                    index_name=index_name,
                    graph_data=graph_data,
                    metadata=metadata,
                )
                print("HybridRetrievalEngine: Saved knowledge graph")
            except Exception as e:
                print(f"HybridRetrievalEngine: Failed to save graph: {e}")

        # 4. Save RAPTOR trees
        if trees:
            try:
                persistence.save_trees(
                    index_name=index_name,
                    trees=trees,
                    metadata=metadata,
                )
                print("HybridRetrievalEngine: Saved document trees")
            except Exception as e:
                print(f"HybridRetrievalEngine: Failed to save trees: {e}")

        return True

    def load_index(self, index_name: str = "default", on_progress: Callable[[str, int, int, str | None], None] | None = None) -> bool:
        """Load index from disk if valid for current corpus/config."""
        persistence = self._get_persistence()
        if not persistence:
            return False

        corpus_version, config_hash = self._compute_versions()

        # Check if saved index is valid
        if not persistence.is_valid(index_name, corpus_version, config_hash):
            print("HybridRetrievalEngine: Saved index invalid or stale, will rebuild")
            return False

        # 1. Load dense index
        try:
            embeddings, chunks, metadata = persistence.load_dense_index(index_name)
            if embeddings is None or chunks is None:
                return False

            with self._index_lock:
                self._embeddings = embeddings
                self._chunks = chunks
                self._corpus_version = corpus_version
                self._config_hash = config_hash
            print(f"HybridRetrievalEngine: Loaded dense index ({len(chunks)} chunks)")
        except Exception as e:
            print(f"HybridRetrievalEngine: Failed to load dense index: {e}")
            return False

        # 2. Load sparse index
        try:
            bm25, tokenized, _ = persistence.load_sparse_index(index_name)
            if bm25 and tokenized:
                with self._index_lock:
                    self._bm25 = bm25
                    self._tokenized_corpus = tokenized
                print("HybridRetrievalEngine: Loaded sparse index")
                if len(self._tokenized_corpus) != len(self._chunks):
                    print("HybridRetrievalEngine: Sparse index size mismatch; rebuilding")
                    with self._index_lock:
                        self._tokenized_corpus = [self._tokenize(c.text) for _, c in self._chunks]
                        BM25Class = _get_bm25()
                        self._bm25 = BM25Class(self._tokenized_corpus) if BM25Class else None
        except Exception as e:
            print(f"HybridRetrievalEngine: Failed to load sparse index: {e}")
            # Rebuild BM25 if load fails but chunks exist
            if self._chunks:
                with self._index_lock:
                    self._tokenized_corpus = [self._tokenize(c.text) for _, c in self._chunks]
                    BM25Class = _get_bm25()
                    if BM25Class:
                        self._bm25 = BM25Class(self._tokenized_corpus)

        # 3. Load GraphRAG data
        if self._config.graph:
            try:
                graph_data, _ = persistence.load_graph(index_name)
                if graph_data:
                    from .graph_rag import GraphRAG
                    with self._index_lock:
                        self._graph_rag = GraphRAG.from_dict(graph_data)
                        self._graph_ready = True
                    print("HybridRetrievalEngine: Loaded knowledge graph")
            except Exception as e:
                print(f"HybridRetrievalEngine: Failed to load graph: {e}")

        # 4. Load RAPTOR trees
        if self._config.raptor or self._config.chunking_strategy != ChunkingStrategy.FIXED:
            try:
                trees_data, _ = persistence.load_trees(index_name)
                if trees_data:
                    from .hierarchy import DocumentTree
                    with self._index_lock:
                        self._trees = {id: DocumentTree.from_dict(d) for id, d in trees_data.items()}
                    print(f"HybridRetrievalEngine: Loaded {len(self._trees)} document trees")
                else:
                    # Rebuild trees in parallel if missing but config says we need them
                    self._rebuild_trees_parallel()
            except Exception as e:
                print(f"HybridRetrievalEngine: Failed to load trees: {e}")
                self._rebuild_trees_parallel()

        docs = self._docs.list()
        with self._index_lock:
            self._doc_timestamps = {}
            for doc in docs:
                ts = self._extract_doc_timestamp(doc.metadata or {})
                if ts:
                    self._doc_timestamps[doc.id] = ts
            self._build_field_tokens(docs)

        return True

    def _rebuild_trees_parallel(self, on_progress: Callable[[str, int, int, str | None], None] | None = None) -> None:
        """Rebuild document trees in parallel."""
        docs = self._docs.list()
        if not docs or not self._chunks:
            return

        from .hierarchy import HierarchyBuilder
        hb = HierarchyBuilder()
        print(f"HybridRetrievalEngine: Rebuilding {len(docs)} trees in parallel...")

        def process_doc(idx_doc):
            idx, doc = idx_doc
            if on_progress:
                on_progress("building_hierarchy", idx, len(docs))
            tree = hb.build(doc.text, doc.id, doc.title)
            doc_chunks = [c for did, c in self._chunks if did == doc.id]
            hb.associate_chunks(tree, doc_chunks, doc.text)
            return doc.id, tree

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Enumerate to track progress
            results = list(executor.map(process_doc, enumerate(docs)))
            with self._index_lock:
                for doc_id, tree in results:
                    self._trees[doc_id] = tree

        if on_progress:
            on_progress("building_hierarchy", len(docs), len(docs))

    def _dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Dense vector similarity search."""
        with self._index_lock:
            embedder = self._embedder
            embeddings = self._embeddings
        if embedder is None or embeddings is None:
            self._last_cache = {"embedding_cache": "skipped"}
            return []
        cached = self._cache_manager.embeddings.get(query)
        if cached is not None:
            query_embedding = np.array(cached)
            self._last_cache = {"embedding_cache": "hit"}
        else:
            query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
            self._cache_manager.embeddings.set(query, query_embedding.tolist())
            self._last_cache = {"embedding_cache": "miss"}

        # Cosine similarity
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def _sparse_search(
        self,
        query: str,
        top_k: int,
        title_boost: float = 0.0,
        heading_boost: float = 0.0,
        proximity_weight: float = 0.0,
    ) -> list[tuple[int, float]]:
        """BM25 keyword search."""
        with self._index_lock:
            if self._bm25 is None:
                return self._fallback_sparse_search(
                    query,
                    top_k,
                    title_boost=title_boost,
                    heading_boost=heading_boost,
                    proximity_weight=proximity_weight,
                )

            query_tokens = self._tokenize(query)
            scores = self._bm25.get_scores(query_tokens)
            query_terms = {t for t in self._tokenize_terms(query) if len(t) > 2}
            top_indices = np.argsort(scores)[::-1][:top_k]
            adjusted: list[tuple[int, float]] = []
            for idx in top_indices:
                base_score = float(scores[idx])
                if base_score <= 0:
                    continue
                base_score += self._field_boost(
                    int(idx),
                    query_terms,
                    title_boost=title_boost,
                    heading_boost=heading_boost,
                )
                if proximity_weight > 0 and query_terms:
                    chunk_idx = int(idx)
                    if chunk_idx < len(self._chunks):
                        chunk_terms = self._tokenize_terms(self._chunks[chunk_idx][1].text)
                        base_score += proximity_weight * self._term_proximity(query_terms, chunk_terms)
                adjusted.append((int(idx), base_score))
            return adjusted

    def _fallback_sparse_search(
        self,
        query: str,
        top_k: int,
        title_boost: float = 0.0,
        heading_boost: float = 0.0,
        proximity_weight: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Fallback sparse search using token overlap when BM25 isn't available."""
        with self._index_lock:
            if not self._tokenized_corpus:
                return []
            query_terms = {t for t in self._tokenize_terms(query) if len(t) > 2}
            query_tokens = set(self._tokenize(query))
            if not query_tokens:
                return []
            scores = []
            for idx, tokens in enumerate(self._tokenized_corpus):
                overlap = query_tokens.intersection(tokens)
                if overlap:
                    base_score = float(len(overlap))
                    base_score += self._field_boost(
                        idx,
                        query_terms,
                        title_boost=title_boost,
                        heading_boost=heading_boost,
                    )
                    if proximity_weight > 0 and query_terms and idx < len(self._chunks):
                        chunk_terms = self._tokenize_terms(self._chunks[idx][1].text)
                        base_score += proximity_weight * self._term_proximity(query_terms, chunk_terms)
                    scores.append((idx, base_score))
            if not scores:
                return []
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

    def get_document_trees(self) -> dict[str, DocumentTree]:
        """Expose built document trees for RAPTOR-style retrieval."""
        with self._index_lock:
            return dict(self._trees)

    def get_keyword_graph_index(self) -> dict[str, set[int]]:
        """Expose keyword graph index (term -> chunk indices)."""
        with self._index_lock:
            return {
                term: set(indices) for term, indices in self._graph.items()
            } if self._graph else {}

    def get_chunk_records(self) -> list[tuple[str, Chunk]]:
        """Return chunk records for downstream multi-resolution retrieval."""
        with self._index_lock:
            return list(self._chunks)

    def _literal_search(self, query: str, top_k: int) -> tuple[list[tuple[int, float]], int]:
        """Literal grep-style scoring for exact phrase/token matches."""
        with self._index_lock:
            if not self._chunks:
                return [], 0
            cleaned = query.strip().lower()
            if not cleaned:
                return [], 0
            tokens = [t for t in self._tokenize(cleaned) if len(t) > 2]
            scores: list[tuple[int, float]] = []
            for idx, (_, chunk) in enumerate(self._chunks):
                text = chunk.text.lower()
                score = 0.0
                if cleaned and cleaned in text:
                    score += 5.0 + text.count(cleaned)
                if tokens:
                    score += sum(1.0 for token in tokens if token in text)
                if score > 0:
                    scores.append((idx, score))
            if not scores:
                return [], 0
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k], len(scores)

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[tuple[int, float]]],
        k: int = 60,
        weights: list[float] | None = None,
    ) -> list[tuple[int, float]]:
        """Combine multiple result lists using RRF.

        RRF score = sum(1 / (k + rank_i)) for each result list
        """
        scores: dict[int, float] = {}

        weight_list = weights or [1.0] * len(result_lists)
        for results, weight in zip(result_lists, weight_list, strict=False):
            for rank, (idx, _) in enumerate(results):
                if idx not in scores:
                    scores[idx] = 0.0
                scores[idx] += weight / (k + rank + 1)

        # Sort by RRF score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def _rerank(self, query: str, candidates: list[tuple[int, float]], top_k: int) -> list[tuple[int, float]]:
        """Rerank candidates using cross-encoder."""
        if self._reranker is None or not candidates:
            return candidates[:top_k]

        # Prepare query-document pairs
        pairs = []
        with self._index_lock:
            for idx, _ in candidates:
                if idx < len(self._chunks):
                    _, chunk = self._chunks[idx]
                    pairs.append((query, chunk.text))

        if not pairs:
            return candidates[:top_k]

        # Get cross-encoder scores
        try:
            scores = self._reranker.predict(pairs)

            # Combine with indices
            reranked = [(candidates[i][0], float(scores[i])) for i in range(len(scores))]
            reranked.sort(key=lambda x: x[1], reverse=True)

            return reranked[:top_k]
        except Exception as e:
            print(f"Warning: Reranking failed: {e}")
            return candidates[:top_k]

    def _apply_colbert_rerank(
        self,
        query: str,
        candidates: list[tuple[int, float]],
        top_k: int,
    ) -> list[tuple[int, float]]:
        """Apply lightweight ColBERT-style late interaction reranking."""
        if not candidates or not self._config.use_colbert:
            return candidates[:top_k]
        if not self._ensure_colbert_ready():
            return candidates[:top_k]
        try:
            query_embed = self._encode_colbert(query)
        except Exception as exc:
            print(f"Warning: ColBERT query encoding failed: {exc}")
            return candidates[:top_k]

        scored: list[tuple[int, float]] = []
        limit = min(len(candidates), self._config.colbert_top_k)
        with self._index_lock:
            chunk_texts = [
                (idx, self._chunks[idx][1].text)
                for idx, _ in candidates[:limit]
                if idx < len(self._chunks)
            ]
        for idx, text in chunk_texts:
            try:
                chunk_embed = self._encode_colbert(text)
                score = self._colbert_similarity(query_embed, chunk_embed)
                scored.append((idx, score))
            except Exception as exc:
                print(f"Warning: ColBERT chunk encoding failed: {exc}")
                continue
        if not scored:
            return candidates[:top_k]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Merge ColBERT ordering with original candidate scores
        seen = set()
        refined: list[tuple[int, float]] = []
        for idx, score in scored:
            refined.append((idx, score))
            seen.add(idx)
            if len(refined) >= top_k:
                break
        if len(refined) < top_k:
            for idx, score in candidates:
                if idx not in seen:
                    refined.append((idx, score))
                if len(refined) >= top_k:
                    break
        self._last_colbert_stats = {
            "applied": True,
            "candidates": len(candidates),
            "scored": len(scored),
            "model": self._config.colbert_model,
        }
        return refined[:top_k]

    def _ensure_colbert_ready(self) -> bool:
        """Lazy-load ColBERT model/tokenizer if enabled."""
        if not self._config.use_colbert or self._colbert_failed:
            return False
        if self._colbert_model is not None and self._colbert_tokenizer is not None:
            return True
        if AutoTokenizer is None or AutoModel is None or torch is None or F is None:
            self._colbert_failed = True
            print("Warning: transformers/torch not available. ColBERT disabled.")
            return False
        try:
            self._colbert_tokenizer = AutoTokenizer.from_pretrained(self._config.colbert_model)
            self._colbert_model = AutoModel.from_pretrained(self._config.colbert_model)
            self._colbert_model.eval()
            if torch.cuda.is_available():
                self._colbert_device = "cuda"
                self._colbert_model.to(self._colbert_device)
            return True
        except Exception as exc:
            print(f"Warning: Failed to load ColBERT model '{self._config.colbert_model}': {exc}")
            self._colbert_failed = True
            return False

    def _encode_colbert(self, text: str):
        """Encode text into token embeddings for ColBERT scoring."""
        if self._colbert_tokenizer is None or self._colbert_model is None or torch is None or F is None:
            raise RuntimeError("ColBERT model not available")
        inputs = self._colbert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self._colbert_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._colbert_model(**inputs)
        token_embeddings = outputs.last_hidden_state.squeeze(0)
        mask = inputs["attention_mask"].squeeze(0).bool()
        valid_embeddings = token_embeddings[mask]
        normalized = F.normalize(valid_embeddings, p=2, dim=-1)
        return normalized

    def _colbert_similarity(self, query_embed, passage_embed) -> float:
        """Compute ColBERT-style MaxSim similarity."""
        if query_embed.shape[0] == 0 or passage_embed.shape[0] == 0:
            return -1.0
        # Query tokens attend over passage tokens
        scores = torch.matmul(query_embed, passage_embed.T)
        max_scores, _ = torch.max(scores, dim=1)
        return float(max_scores.mean().item())

    def set_precision_mode(self, enable_colbert: bool) -> None:
        """Enable/disable late-interaction reranking dynamically."""
        enable = bool(enable_colbert)
        if self._config.use_colbert == enable:
            if enable:
                self._ensure_colbert_ready()
            return
        self._config.use_colbert = enable
        if enable:
            self._ensure_colbert_ready()

    def colbert_enabled(self) -> bool:
        """Return True if ColBERT precision mode is active."""
        return bool(self._config.use_colbert and self._colbert_model is not None)

    def precision_stats(self) -> dict[str, Any]:
        """Return latest ColBERT precision stats for telemetry."""
        return dict(self._last_colbert_stats)

    def set_rerank_mode(self, enabled: bool) -> None:
        """Enable/disable cross-encoder reranking on the fly."""
        if not enabled:
            self._config.use_reranking = False
            return
        self._config.use_reranking = bool(self._reranker)

    async def query(
        self,
        text: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        routing_params: dict[str, Any] | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> list[RetrievalResult]:
        """Execute hybrid retrieval query.

        1. Dense search for semantic matches
        2. Sparse BM25 search for keyword matches
        3. RRF fusion to combine results
        4. Cross-encoder reranking for precision
        """
        def emit(msg: str, val: float) -> None:
            if on_progress:
                on_progress(msg, val)

        if not self._chunks:
            emit("Building retrieval index...", 0.0)
            self.build()

        if not self._chunks or not text.strip():
            return []

        # Reset cache info for this query
        self._last_cache = {}
        overrides = self._normalize_routing_params(routing_params)
        # Get candidates from both methods in parallel
        rerank_k = self._config.rerank_top_k
        loop = asyncio.get_event_loop()

        async def run_dense():
            emit("Semantic search (dense)...", 0.1)
            return await loop.run_in_executor(None, self._dense_search, text, rerank_k)

        async def run_sparse():
            emit("Keyword search (sparse)...", 0.3)
            return await loop.run_in_executor(None, self._sparse_search, text, rerank_k,
                                           overrides["title_boost"], overrides["heading_boost"],
                                           overrides.get("proximity_weight", 0.0))

        async def run_literal():
            emit("Literal search...", 0.4)
            return await loop.run_in_executor(None, self._literal_search, text, rerank_k)

        dense_task = asyncio.create_task(run_dense())
        sparse_task = asyncio.create_task(run_sparse())
        literal_task = asyncio.create_task(run_literal())

        dense_results, sparse_results, (literal_results, literal_hits) = await asyncio.gather(
            dense_task, sparse_task, literal_task
        )

        if literal_hits:
            self._last_cache["literal_hits"] = literal_hits

        # Handle fallback cases
        if not dense_results and not sparse_results and not literal_results:
            return []

        # Combine with RRF
        emit("Fusing results...", 0.5)
        result_lists = [lst for lst in (dense_results, sparse_results, literal_results) if lst]
        if len(result_lists) > 1:
            weights: list[float] = []
            for result_list in result_lists:
                if result_list is dense_results:
                    weights.append(overrides["dense_weight"])
                elif result_list is sparse_results:
                    weights.append(overrides["sparse_weight"])
                else:
                    weights.append(overrides["literal_weight"])
            fused = self._reciprocal_rank_fusion(result_lists, weights=weights)
        else:
            fused = result_lists[0]

        # Rerank if enabled (Slowest step, run in executor)
        if self._config.use_reranking and self._reranker:
            emit("Reranking candidates...", 0.6)
            final_results = await loop.run_in_executor(None, self._rerank, text, fused, top_k)
        else:
            final_results = fused[:top_k]

        self._last_colbert_stats = {
            "applied": False,
            "candidates": len(final_results),
            "scored": 0,
        }
        if self._config.use_colbert:
            emit("Applying ColBERT rerank...", 0.8)
            final_results = await loop.run_in_executor(None, self._apply_colbert_rerank, text, final_results, top_k)

        emit("Finalizing results...", 0.9)
        final_results = self._apply_recency_boost(
            final_results,
            recency_weight=overrides["recency_weight"],
            half_life_days=overrides["recency_half_life_days"],
        )
        final_results.sort(key=lambda x: x[1], reverse=True)
        final_results = await loop.run_in_executor(
            None,
            self._apply_diversity_rerank,
            final_results,
            top_k,
            overrides["diversity"]
        )

        # Build result objects
        id_to_doc = {doc.id: doc for doc in self._docs.list()}
        results: list[RetrievalResult] = []
        allowed_ids = set(document_ids) if document_ids else None

        with self._index_lock:
            # Hierarchy/Graph Expansion (Phase 4/6)
            expanded_indices = {idx for idx, _ in final_results}

            # 1. Graph expansion: Find related chunks via shared keywords
            if self._config.graph and allowed_ids is None:
                query_tokens = self._tokenize(text)
                for token in query_tokens:
                    if token in self._graph:
                        # Add top related chunks for each keyword
                        expanded_indices.update(list(self._graph[token])[:2])

            retrieval_method = "hybrid" if len(result_lists) > 1 else (
                "dense" if dense_results else "sparse" if sparse_results else "literal"
            )

            for idx, score in final_results:
                if idx < len(self._chunks):
                    doc_id, chunk = self._chunks[idx]
                    if allowed_ids is not None and doc_id not in allowed_ids:
                        continue
                    doc = id_to_doc.get(doc_id)
                    if not doc:
                        continue

                    # 2. Hierarchy expansion (RAPTOR): Get parent summary
                    extra_context = []
                    if self._config.raptor and doc_id in self._trees:
                        from .hierarchy import HierarchicalRetriever
                        hr = HierarchicalRetriever(self._trees[doc_id])
                        extra_context = hr.get_context_chain(f"{chunk.index}")

                    # Create a transient document for the chunk
                    chunk_text = chunk.text
                    if extra_context:
                        # Prepend hierarchical context to chunk text
                        context_str = "\n".join(extra_context)
                        chunk_text = f"[Hierarchy Context]\n{context_str}\n\n[Chunk Content]\n{chunk.text}"

                    chunk_doc = Document(
                        id=f"{doc.id}-{chunk.index}",
                        title=doc.title,
                        text=chunk_text,
                        metadata=doc.metadata,
                    )
                    results.append(RetrievalResult(
                        document=chunk_doc,
                        score=score,
                        chunk_text=chunk.text,
                        retrieval_method=retrieval_method,
                        chunk_id=f"{doc.id}-{chunk.index}",
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                    ))

        return results

    def get_last_cache_info(self) -> dict[str, Any]:
        return dict(self._last_cache)

    def get_model_status(self) -> dict[str, Any]:
        """Get loading status of internal models (G1)."""
        status = {
            "deployment_profile": self._config.deployment_profile,
            "backend_map": dict(self._config.backend_map),
            "embedding_model": {
                "name": self._config.embedding_model,
                "status": "ready" if self._embedder else ("failed" if self._embedder_failed else "pending"),
                "is_local": os.path.isdir(self._config.embedding_model) if self._config.embedding_model else False,
                "backend_id": self._config.backend_map.get("embedding", ""),
            },
            "reranker_model": {
                "name": self._config.reranker_model,
                "status": "ready" if self._reranker else ("failed" if self._reranker_failed else "pending"),
                "is_local": os.path.isdir(self._config.reranker_model) if self._config.reranker_model else False,
                "backend_id": self._config.backend_map.get("reranker", ""),
            }
        }
        return status

    def get_readiness_snapshot(self) -> dict[str, Any]:
        """Return non-invasive retrieval readiness facts."""
        docs = self._docs.list()
        chunk_count = len(self._chunks)
        embedding_count = int(len(self._embeddings)) if self._embeddings is not None else 0
        sparse_ready = bool(self._bm25 and self._tokenized_corpus)
        dense_ready = self._embeddings is not None and embedding_count > 0
        model_status = self.get_model_status()
        bq_enabled = bool(getattr(self, "_bq_enabled", False))
        bq_ready = bool(getattr(self, "_bq_ready", False))

        return {
            "document_count": len(docs),
            "chunk_count": chunk_count,
            "embedding_count": embedding_count,
            "sparse_ready": sparse_ready,
            "dense_ready": dense_ready,
            "bq_enabled": bq_enabled,
            "bq_ready": bq_ready,
            "index_ready": len(docs) == 0 or chunk_count > 0,
            "model_status": model_status,
            "config": {
                "embedding_model": self._config.embedding_model,
                "reranker_model": self._config.reranker_model,
                "use_reranking": self._config.use_reranking,
                "use_colbert": self._config.use_colbert,
                "raptor": self._config.raptor,
                "graph": self._config.graph,
                "deployment_profile": self._config.deployment_profile,
            },
        }

    def get_corpus_manifest(self) -> dict[str, Any]:
        """Return a content-safe corpus manifest for eval reproducibility."""
        documents = sorted(self._docs.list(), key=lambda doc: doc.id)
        document_hashes = []
        secret_terms = ("apikey", "authorization", "token", "secret", "password", "credential")
        for doc in documents:
            metadata = {
                key: value
                for key, value in sorted((doc.metadata or {}).items())
                if not any(term in re.sub(r"[^a-z0-9]", "", key.lower()) for term in secret_terms)
            }
            payload = {
                "id": doc.id,
                "title": doc.title,
                "text_sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
                "metadata": metadata,
            }
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            document_hashes.append({"id": doc.id, "title": doc.title, "sha256": digest})

        fingerprint_payload = {
            "documents": document_hashes,
            "chunk_count": len(self._chunks),
            "embedding_count": int(len(self._embeddings)) if self._embeddings is not None else 0,
            "corpus_version": self.get_corpus_version(),
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return {
            "fingerprint": fingerprint,
            "document_count": len(documents),
            "chunk_count": len(self._chunks),
            "embedding_count": int(len(self._embeddings)) if self._embeddings is not None else 0,
            "corpus_version": self.get_corpus_version(),
            "documents": document_hashes,
        }

    def get_runtime_profile(self) -> dict[str, Any]:
        return {
            "deployment_profile": self._config.deployment_profile,
            "backends": dict(self._config.backend_map),
            "fallbacks": dict(self._config.fallback_map),
            "embedding_model": self._config.embedding_model,
            "reranker_model": self._config.reranker_model,
            "chunking_strategy": self._config.chunking_strategy.value,
        }


# Backward compatibility: alias for the old class name
class RetrievalEngine(HybridRetrievalEngine):
    """Alias for backward compatibility with existing code."""

    def __init__(self, documents: DocumentStore) -> None:
        # Use a simpler config for backward compat
        config = HybridConfig(
            use_reranking=False,  # Don't require reranking for basic usage
            chunking_strategy=ChunkingStrategy.FIXED,  # Original behavior
        )
        super().__init__(documents, config)


# Keep original RetrievalResult available
__all__ = [
    "RetrievalResult",
    "HybridRetrievalEngine",
    "RetrievalEngine",
    "HybridConfig",
]
