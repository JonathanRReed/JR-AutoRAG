from __future__ import annotations

from enum import Enum

from pydantic import AnyHttpUrl, BaseModel, field_validator


class ProviderConfig(BaseModel):
    name: str
    base_url: AnyHttpUrl
    planner_model: str | None = None
    gatherer_model: str | None = None
    generator_model: str | None = None
    api_key: str | None = None


class ProviderProfile(BaseModel):
    name: str
    provider: ProviderConfig


class RetrievalDefaults(BaseModel):
    """Retrieval configuration optimized for hybrid retrieval.

    Presets:
    - Fast: dense_k=3, target_tokens=800, coverage_target=0.5
    - Balanced (default): dense_k=5, target_tokens=1600, coverage_target=0.7
    - Thorough: dense_k=10, target_tokens=3000, coverage_target=0.9
    """
    hybrid: bool = True  # Enable hybrid search (dense + BM25)
    dense_k: int = 5  # Top chunks from dense retrieval
    sparse_k: int = 10  # Top chunks from BM25 retrieval
    dense_weight: float = 0.6  # RRF weight for dense results
    sparse_weight: float = 0.4  # RRF weight for sparse results
    rerank_pool: int = 20  # Candidates for reranking
    top_n: int = 5  # Final chunks to use in context
    compression: bool = False  # Context compression (requires LLM)
    target_tokens: int = 1600  # Reasonable for local LLMs
    raptor: bool = False  # Hierarchical indexing (future)
    graph: bool = False  # Graph retrieval (future)
    coverage_target: float = 0.7  # Target 70% coverage
    max_context_tokens: int = 4096  # Safe default for most local models
    
    # New hybrid retrieval options
    chunking_strategy: str = "semantic"  # "fixed", "semantic", or "recursive"
    embedding_model: str = "BAAI/bge-base-en-v1.5"  # Sentence transformer model
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder
    use_reranking: bool = True  # Enable cross-encoder reranking
    use_colbert: bool = False  # Enable ColBERT late-interaction reranking
    colbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    colbert_top_k: int = 12
    chunk_size: int = 400  # Target chunk size in characters
    chunk_overlap: int = 50  # Overlap between chunks
    planner_mode: str = "smart"  # "simple" or "smart"
    flare_generation: bool = True  # Enable FLARE mid-generation retrieval
    enforce_evidence_contract: bool = True  # Require evidence-first self-checks
    multi_resolution: bool = True  # Enable parent-child context expansion
    recency_weight: float = 0.1  # Recency prior boost for newer docs
    recency_half_life_days: float = 90.0  # Days for recency score to halve
    title_boost: float = 0.6  # Field-aware boost for title matches
    heading_boost: float = 0.4  # Field-aware boost for heading matches
    proximity_weight: float = 0.5  # Term-proximity boost for BM25
    diversity: float = 0.0  # 0-1: prefer diverse chunks when >0
    use_hyde: bool = False  # Enable HyDE (Hypothetical Document Embeddings)
    abstain_when_unverified: bool = False  # Abstain when evidence is insufficient
    self_rag_critic: bool = False  # Enable Self-RAG LLM-based critic (v2.0)

    @field_validator("raptor", mode="before")
    @classmethod
    def _coerce_raptor(cls, value: bool | str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"on", "true", "yes", "enabled", "1"}:
                return True
            if normalized in {"off", "false", "no", "disabled", "0"}:
                return False
        return bool(value)


# Retrieval presets for different use cases (5-tier system)
# Speed ←→ Accuracy spectrum: turbo → fast → balanced → thorough → ultra_accurate
RETRIEVAL_PRESETS = {
    "turbo": RetrievalDefaults(
        hybrid=True,
        dense_k=2,
        sparse_k=3,
        rerank_pool=8,
        top_n=2,
        target_tokens=400,
        coverage_target=0.3,
        max_context_tokens=2048,
        use_reranking=False,  # Skip reranking for max speed
        chunking_strategy="fixed",
        chunk_size=800,
        flare_generation=False,
        enforce_evidence_contract=False,
        multi_resolution=False,
        raptor=False,
        graph=False,
    ),
    "fast": RetrievalDefaults(
        hybrid=True,
        dense_k=3,
        sparse_k=5,
        rerank_pool=12,
        top_n=3,
        target_tokens=800,
        coverage_target=0.5,
        max_context_tokens=2048,
        use_reranking=True,
        chunking_strategy="fixed",
        chunk_size=600,
        flare_generation=False,
        enforce_evidence_contract=False,
        multi_resolution=False,
        raptor=False,
        graph=False,
    ),
    "balanced": RetrievalDefaults(
        hybrid=True,
        dense_k=5,
        sparse_k=10,
        rerank_pool=20,
        top_n=5,
        target_tokens=1600,
        coverage_target=0.7,
        max_context_tokens=4096,
        use_reranking=True,
        chunking_strategy="semantic",
        flare_generation=False,
        enforce_evidence_contract=False,
        multi_resolution=True,
        raptor=False,
        graph=False,
        abstain_when_unverified=False,  # v2.0: DISABLED - coverage calc bug
    ),
    "thorough": RetrievalDefaults(
        hybrid=True,
        dense_k=10,
        sparse_k=20,
        rerank_pool=40,
        top_n=8,
        target_tokens=3000,
        coverage_target=0.9,
        max_context_tokens=8192,
        use_reranking=True,
        chunking_strategy="semantic",
        chunk_size=300,
        chunk_overlap=75,
        flare_generation=True,
        enforce_evidence_contract=False,
        multi_resolution=True,
        raptor=True,
        graph=False,
        use_hyde=True,
        abstain_when_unverified=False,  # v2.0: DISABLED - coverage calc bug
        self_rag_critic=False,  # v2.0: DISABLED - causes late failures
    ),
    "ultra_accurate": RetrievalDefaults(
        hybrid=True,
        dense_k=20,
        sparse_k=30,
        rerank_pool=50,
        top_n=12,
        target_tokens=5000,
        coverage_target=0.95,
        max_context_tokens=16384,
        use_reranking=True,
        chunking_strategy="semantic",
        chunk_size=250,
        chunk_overlap=100,
        flare_generation=True,
        enforce_evidence_contract=True,
        multi_resolution=True,
        raptor=True,
        graph=True,
        diversity=0.3,
        use_hyde=True,
        abstain_when_unverified=False,  # v2.0: DISABLED - coverage calc bug
        self_rag_critic=False,  # v2.0: DISABLED - causes late failures
    ),
}


class StageBudgetDefaults(BaseModel):
    """Per-stage timeout and token budget configuration (P0.2)."""
    
    # Timeouts (milliseconds)
    planner_timeout_ms: int = 3000
    gatherer_timeout_ms: int = 12000
    rerank_timeout_ms: int = 5000
    compression_timeout_ms: int = 4000
    generation_timeout_ms: int = 20000
    verification_timeout_ms: int = 5000
    total_timeout_ms: int = 60000
    
    # Token budgets
    retrieval_token_budget: int = 8000
    rerank_pool_budget: int = 50
    compression_token_budget: int = 4000
    answer_token_budget: int = 2000


class AppConfig(BaseModel):
    profile: str = "Default"
    provider: ProviderConfig | None = None
    provider_profiles: list[ProviderProfile] = []
    retrieval: RetrievalDefaults = RetrievalDefaults()
    
    # P0.1: Query mode switch (grounded = docs only, open_domain = LLM can use knowledge)
    query_mode: str = "grounded"
    
    # P0.2: Stage budgets for timeouts and token limits
    stage_budgets: StageBudgetDefaults = StageBudgetDefaults()


class ProviderKind(str, Enum):
    OLLAMA = "ollama"
    OLLAMA_CLOUD = "ollama_cloud"
    LM_STUDIO = "lmstudio"
    OPENAI_COMPAT = "openai"
    OPENROUTER = "openrouter"


class LocalProviderInfo(BaseModel):
    kind: ProviderKind
    name: str
    base_url: str
    models: list[str] = []
    running: list[str] = []
    version: str | None = None
    status: str = "ok"
    error_message: str | None = None


class ModelStatusRequest(BaseModel):
    embedding_model: str | None = None
    reranker_model: str | None = None


class ModelStatusResponse(BaseModel):
    embedding: str
    reranker: str
    embedding_message: str | None = None
    reranker_message: str | None = None


class ModelDownloadRequest(BaseModel):
    kind: str
    model: str
