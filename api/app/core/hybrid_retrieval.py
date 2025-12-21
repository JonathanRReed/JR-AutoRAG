"""Hybrid retrieval engine with dense embeddings, BM25, and reranking.

This module provides a state-of-the-art retrieval implementation that combines:
- Dense vector search using sentence-transformers
- BM25 sparse retrieval for keyword matching
- Reciprocal Rank Fusion (RRF) for combining results
- Cross-encoder reranking for precision
"""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from .cache import get_cache_manager
from .documents import Document, DocumentStore
from .chunking import Chunk, ChunkingStrategy, get_chunker
from .hierarchy import DocumentTree

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    from transformers import AutoTokenizer, AutoModel
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



@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""
    # Model configuration
    embedding_model: str = "BAAI/bge-base-en-v1.5"
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
        
        # Persistence
        self._index_persistence = None
        self._corpus_version: str = ""
        self._config_hash: str = ""
        
        # Phase 4/6: Hierarchical & Graph structures
        self._trees: dict[str, Any] = {}  # doc_id -> DocumentTree
        self._graph: dict[str, set[int]] = {}  # term -> set of chunk indices
        
        # Load models
        self._init_models()
    
    def _init_models(self) -> None:
        """Lazy load heavy models."""
        if self._config.embedding_model and not self._embedder_failed and self._embedder is None:
            SentenceTransformer = _get_sentence_transformer()
            if SentenceTransformer:
                try:
                    print(f"Loading embedding model: {self._config.embedding_model}...")
                    try:
                        self._embedder = SentenceTransformer(
                            self._config.embedding_model,
                            device="cpu",
                            model_kwargs={"low_cpu_mem_usage": False, "device_map": None},
                        )
                    except TypeError:
                        self._embedder = SentenceTransformer(self._config.embedding_model, device="cpu")
                    print("Embedding model loaded successfully.")
                except Exception as e:
                    print(f"❌ Error: Could not load embedding model: {e}")
                    self._embedder = None
                    self._embedder_failed = True
            else:
                print("⚠️ Warning: sentence-transformers not installed. Dense retrieval disabled.")
        
        if self._config.use_reranking and self._config.reranker_model and not self._reranker_failed and self._reranker is None:
            CrossEncoder = _get_cross_encoder()
            if CrossEncoder:
                try:
                    print(f"Loading reranker model: {self._config.reranker_model}...")
                    try:
                        self._reranker = CrossEncoder(
                            self._config.reranker_model,
                            device="cpu",
                            automodel_args={"low_cpu_mem_usage": False, "device_map": None},
                        )
                    except TypeError:
                        self._reranker = CrossEncoder(self._config.reranker_model, device="cpu")
                    print("Reranker model loaded successfully.")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load reranker model: {e}")
                    self._reranker = None
                    self._reranker_failed = True
    
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
                return datetime.fromtimestamp(epoch, tz=timezone.utc)
            except Exception:
                return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
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
        needed = {t for t in query_terms}
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
        now = datetime.now(timezone.utc)
        adjusted: list[tuple[int, float]] = []
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
    
    def build(self) -> None:
        """Build the retrieval index from all documents."""
        docs = self._docs.list()
        print(f"HybridRetrievalEngine: Building index for {len(docs)} documents...")
        
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
        
        for doc in docs:
            if not doc.text.strip():
                print(f"HybridRetrievalEngine: Skipping empty document: {doc.title}")
                continue
                
            chunks = chunker.chunk(doc.text)
            print(f"HybridRetrievalEngine: Processing '{doc.title}' -> {len(chunks)} chunks")
            for chunk in chunks:
                self._chunks.append((doc.id, chunk))
                corpus_texts.append(chunk.text)
        
        if not corpus_texts:
            print("HybridRetrievalEngine: No text chunks generated (all docs might be empty).")
            return
        
        # Build dense embeddings
        if self._embedder:
            try:
                self._embeddings = self._embedder.encode(
                    corpus_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except Exception as e:
                print(f"Warning: Embedding failed: {e}")
                self._embeddings = None
        
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

        # Build hierarchy and graph if enabled
        if self._config.raptor or self._config.chunking_strategy != ChunkingStrategy.FIXED:
            from .hierarchy import HierarchyBuilder
            hb = HierarchyBuilder()
            for doc in docs:
                tree = hb.build(doc.text, doc.id, doc.title)
                # Associate chunks for this doc
                doc_chunks = [c for did, c in self._chunks if did == doc.id]
                hb.associate_chunks(tree, doc_chunks, doc.text)
                self._trees[doc.id] = tree
        
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

        # Metadata-aware scoring caches
        self._doc_timestamps = {}
        for doc in docs:
            ts = self._extract_doc_timestamp(doc.metadata or {})
            if ts:
                self._doc_timestamps[doc.id] = ts
        self._build_field_tokens(docs)
        
        # Auto-save if persistence enabled
        if self._persist_path and self._chunks:
            self.save_index()
    
    def _get_persistence(self):
        """Lazy-load persistence manager."""
        if self._index_persistence is None and self._persist_path:
            from .persistence import IndexPersistence
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
        
        if not self._chunks or self._embeddings is None:
            print("HybridRetrievalEngine: No index to save")
            return False
        
        from .persistence import IndexMetadata
        import time
        
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
        
        # Save dense index
        try:
            persistence.save_dense_index(
                index_name=index_name,
                embeddings=self._embeddings,
                chunks=self._chunks,
                metadata=metadata,
            )
            print(f"HybridRetrievalEngine: Saved dense index ({len(self._chunks)} chunks)")
        except Exception as e:
            print(f"HybridRetrievalEngine: Failed to save dense index: {e}")
            return False
        
        # Save sparse index if available
        if self._bm25 and self._tokenized_corpus:
            try:
                persistence.save_sparse_index(
                    index_name=index_name,
                    bm25=self._bm25,
                    tokenized_corpus=self._tokenized_corpus,
                    metadata=metadata,
                )
                print(f"HybridRetrievalEngine: Saved sparse index")
            except Exception as e:
                print(f"HybridRetrievalEngine: Failed to save sparse index: {e}")
        
        return True
    
    def load_index(self, index_name: str = "default") -> bool:
        """Load index from disk if valid for current corpus/config."""
        persistence = self._get_persistence()
        if not persistence:
            return False
        
        corpus_version, config_hash = self._compute_versions()
        
        # Check if saved index is valid
        if not persistence.is_valid(index_name, corpus_version, config_hash):
            print("HybridRetrievalEngine: Saved index invalid or stale, will rebuild")
            return False
        
        # Load dense index
        try:
            embeddings, chunks, metadata = persistence.load_dense_index(index_name)
            if embeddings is None or chunks is None:
                return False
            
            self._embeddings = embeddings
            self._chunks = chunks
            self._corpus_version = corpus_version
            self._config_hash = config_hash
            print(f"HybridRetrievalEngine: Loaded dense index ({len(chunks)} chunks)")
        except Exception as e:
            print(f"HybridRetrievalEngine: Failed to load dense index: {e}")
            return False
        
        # Load sparse index
        try:
            bm25, tokenized, _ = persistence.load_sparse_index(index_name)
            if bm25 and tokenized:
                self._bm25 = bm25
                self._tokenized_corpus = tokenized
                print("HybridRetrievalEngine: Loaded sparse index")
        except Exception as e:
            print(f"HybridRetrievalEngine: Failed to load sparse index: {e}")
            # Rebuild BM25 from chunks
            self._tokenized_corpus = [self._tokenize(c.text) for _, c in self._chunks]
            BM25Class = _get_bm25()
            if BM25Class:
                self._bm25 = BM25Class(self._tokenized_corpus)
        docs = self._docs.list()
        if self._config.raptor or self._config.chunking_strategy != ChunkingStrategy.FIXED:
            from .hierarchy import HierarchyBuilder
            hb = HierarchyBuilder()
            self._trees = {}
            for doc in docs:
                tree = hb.build(doc.text, doc.id, doc.title)
                doc_chunks = [c for did, c in self._chunks if did == doc.id]
                hb.associate_chunks(tree, doc_chunks, doc.text)
                self._trees[doc.id] = tree
        self._doc_timestamps = {}
        for doc in docs:
            ts = self._extract_doc_timestamp(doc.metadata or {})
            if ts:
                self._doc_timestamps[doc.id] = ts
        self._build_field_tokens(docs)
        
        return True
    
    def _dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Dense vector similarity search."""
        if self._embedder is None or self._embeddings is None:
            self._last_cache = {"embedding_cache": "skipped"}
            return []
        cached = self._cache_manager.embeddings.get(query)
        if cached is not None:
            query_embedding = np.array(cached)
            self._last_cache = {"embedding_cache": "hit"}
        else:
            query_embedding = self._embedder.encode([query], convert_to_numpy=True)[0]
            self._cache_manager.embeddings.set(query, query_embedding.tolist())
            self._last_cache = {"embedding_cache": "miss"}
        
        # Cosine similarity
        similarities = np.dot(self._embeddings, query_embedding) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def _sparse_search(
        self,
        query: str,
        top_k: int,
        *,
        title_boost: float = 0.0,
        heading_boost: float = 0.0,
        proximity_weight: float = 0.0,
    ) -> list[tuple[int, float]]:
        """BM25 keyword search."""
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
                chunk_terms = self._tokenize_terms(self._chunks[int(idx)][1].text)
                base_score += proximity_weight * self._term_proximity(query_terms, chunk_terms)
            adjusted.append((int(idx), base_score))
        return adjusted

    def _fallback_sparse_search(
        self,
        query: str,
        top_k: int,
        *,
        title_boost: float = 0.0,
        heading_boost: float = 0.0,
        proximity_weight: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Fallback sparse search using token overlap when BM25 isn't available."""
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
                if proximity_weight > 0 and query_terms:
                    chunk_terms = self._tokenize_terms(self._chunks[idx][1].text)
                    base_score += proximity_weight * self._term_proximity(query_terms, chunk_terms)
                scores.append((idx, base_score))
        if not scores:
            return []
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_document_trees(self) -> dict[str, DocumentTree]:
        """Expose built document trees for RAPTOR-style retrieval."""
        return dict(self._trees)

    def get_keyword_graph_index(self) -> dict[str, set[int]]:
        """Expose keyword graph index (term -> chunk indices)."""
        return {
            term: set(indices) for term, indices in self._graph.items()
        } if self._graph else {}

    def get_chunk_records(self) -> list[tuple[str, Chunk]]:
        """Return chunk records for downstream multi-resolution retrieval."""
        return list(self._chunks)

    def _literal_search(self, query: str, top_k: int) -> tuple[list[tuple[int, float]], int]:
        """Literal grep-style scoring for exact phrase/token matches."""
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
        for results, weight in zip(result_lists, weight_list):
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
        for idx, _ in candidates[:limit]:
            if idx >= len(self._chunks):
                continue
            _, chunk = self._chunks[idx]
            try:
                chunk_embed = self._encode_colbert(chunk.text)
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
    
    def query(
        self,
        text: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        routing_params: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Execute hybrid retrieval query.
        
        1. Dense search for semantic matches
        2. Sparse BM25 search for keyword matches  
        3. RRF fusion to combine results
        4. Cross-encoder reranking for precision
        """
        if not self._chunks:
            self.build()
        
        if not self._chunks or not text.strip():
            return []
        
        # Reset cache info for this query
        self._last_cache = {}
        overrides = self._normalize_routing_params(routing_params)
        # Get candidates from both methods
        rerank_k = self._config.rerank_top_k
        dense_results = self._dense_search(text, rerank_k)
        sparse_results = self._sparse_search(
            text,
            rerank_k,
            title_boost=overrides["title_boost"],
            heading_boost=overrides["heading_boost"],
            proximity_weight=overrides["proximity_weight"],
        )
        literal_results, literal_hits = self._literal_search(text, rerank_k)
        if literal_hits:
            self._last_cache["literal_hits"] = literal_hits
        
        # Handle fallback cases
        if not dense_results and not sparse_results and not literal_results:
            return []
        
        # Combine with RRF if both available, otherwise use what we have
        result_lists = [lst for lst in (dense_results, sparse_results, literal_results) if lst]
        if dense_results and sparse_results:
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
        
        # Rerank if enabled
        if self._config.use_reranking and self._reranker:
            final_results = self._rerank(text, fused, top_k)
        else:
            final_results = fused[:top_k]
        
        self._last_colbert_stats = {
            "applied": False,
            "candidates": len(final_results),
            "scored": 0,
        }
        if self._config.use_colbert:
            final_results = self._apply_colbert_rerank(text, final_results, top_k)

        final_results = self._apply_recency_boost(
            final_results,
            recency_weight=overrides["recency_weight"],
            half_life_days=overrides["recency_half_life_days"],
        )
        final_results.sort(key=lambda x: x[1], reverse=True)
        final_results = self._apply_diversity_rerank(
            final_results,
            top_k=top_k,
            diversity=overrides["diversity"],
        )
        
        # Build result objects
        id_to_doc = {doc.id: doc for doc in self._docs.list()}
        results: list[RetrievalResult] = []
        allowed_ids = set(document_ids) if document_ids else None
        
        # Hierarchy/Graph Expansion (Phase 4/6)
        expanded_indices = set(idx for idx, _ in final_results)
        
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
                if not doc: continue
                
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
