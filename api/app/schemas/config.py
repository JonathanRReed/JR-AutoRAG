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
    chunk_size: int = 400  # Target chunk size in characters
    chunk_overlap: int = 50  # Overlap between chunks
    planner_mode: str = "smart"  # "simple" or "smart"

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


# Retrieval presets for different use cases
RETRIEVAL_PRESETS = {
    "fast": RetrievalDefaults(
        hybrid=True,
        dense_k=3,
        sparse_k=5,
        rerank_pool=10,
        top_n=3,
        target_tokens=800,
        coverage_target=0.5,
        max_context_tokens=2048,
        use_reranking=False,  # Skip reranking for speed
        chunking_strategy="fixed",  # Faster chunking
        chunk_size=600,
    ),
    "balanced": RetrievalDefaults(),  # Uses defaults above
    "thorough": RetrievalDefaults(
        hybrid=True,
        dense_k=10,
        sparse_k=20,
        rerank_pool=40,  # More candidates for reranking
        top_n=8,
        target_tokens=3000,
        coverage_target=0.9,
        max_context_tokens=8192,
        use_reranking=True,
        chunking_strategy="semantic",
        chunk_size=300,  # Smaller chunks for precision
        chunk_overlap=75,
    ),
}


class AppConfig(BaseModel):
    profile: str = "Default"
    provider: ProviderConfig | None = None
    provider_profiles: list[ProviderProfile] = []
    retrieval: RetrievalDefaults = RetrievalDefaults()


class ProviderKind(str, Enum):
    OLLAMA = "ollama"
    LM_STUDIO = "lmstudio"
    OPENAI_COMPAT = "openai"


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
