"""Hybrid retrieval engine with dense embeddings, BM25, and reranking.

This module provides a state-of-the-art retrieval implementation that combines:
- Dense vector search using sentence-transformers
- BM25 sparse retrieval for keyword matching
- Reciprocal Rank Fusion (RRF) for combining results
- Cross-encoder reranking for precision
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .cache import get_cache_manager
from .documents import Document, DocumentStore
from .chunking import Chunk, ChunkingStrategy, get_chunker

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer, CrossEncoder

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
    
    # Chunking
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    chunk_size: int = 400
    chunk_overlap: int = 50
    
    # Feature toggles
    raptor: bool = False
    graph: bool = False


class HybridRetrievalEngine:
    """Advanced retrieval engine with hybrid search and reranking.
    
    Combines dense vector search (semantic) with BM25 (keyword) retrieval,
    then optionally applies cross-encoder reranking for maximum precision.
    
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
    
    def _sparse_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """BM25 keyword search."""
        if self._bm25 is None:
            return self._fallback_sparse_search(query, top_k)
        
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def _fallback_sparse_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Fallback sparse search using token overlap when BM25 isn't available."""
        if not self._tokenized_corpus:
            return []
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []
        scores = []
        for idx, tokens in enumerate(self._tokenized_corpus):
            overlap = query_tokens.intersection(tokens)
            if overlap:
                scores.append((idx, float(len(overlap))))
        if not scores:
            return []
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

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
    ) -> list[tuple[int, float]]:
        """Combine multiple result lists using RRF.
        
        RRF score = sum(1 / (k + rank_i)) for each result list
        """
        scores: dict[int, float] = {}
        
        for results in result_lists:
            for rank, (idx, _) in enumerate(results):
                if idx not in scores:
                    scores[idx] = 0.0
                scores[idx] += 1.0 / (k + rank + 1)
        
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
    
    def query(
        self,
        text: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
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
        # Get candidates from both methods
        rerank_k = self._config.rerank_top_k
        dense_results = self._dense_search(text, rerank_k)
        sparse_results = self._sparse_search(text, rerank_k)
        literal_results, literal_hits = self._literal_search(text, rerank_k)
        if literal_hits:
            self._last_cache["literal_hits"] = literal_hits
        
        # Handle fallback cases
        if not dense_results and not sparse_results and not literal_results:
            return []
        
        # Combine with RRF if both available, otherwise use what we have
        result_lists = [lst for lst in (dense_results, sparse_results, literal_results) if lst]
        if len(result_lists) > 1:
            fused = self._reciprocal_rank_fusion(result_lists)
        else:
            fused = result_lists[0]
        
        # Rerank if enabled
        if self._config.use_reranking and self._reranker:
            final_results = self._rerank(text, fused, top_k)
        else:
            final_results = fused[:top_k]
        
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
