from enum import Enum

from pydantic import AnyHttpUrl, BaseModel


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
    """Retrieval configuration optimized for local TF-IDF based retrieval.

    Presets:
    - Fast: dense_k=3, target_tokens=800, coverage_target=0.5
    - Balanced (default): dense_k=5, target_tokens=1600, coverage_target=0.7
    - Thorough: dense_k=10, target_tokens=3000, coverage_target=0.9
    """
    hybrid: bool = False  # TF-IDF only for local, no dense embeddings
    dense_k: int = 5  # Top chunks to retrieve (was 40, too aggressive for TF-IDF)
    sparse_k: int = 10  # Not used in current TF-IDF impl, kept for future hybrid
    rerank_pool: int = 10  # Not used in current impl, kept for future reranking
    top_n: int = 5  # Final chunks to use in context
    compression: bool = False  # No compression for local (requires LLM)
    target_tokens: int = 1600  # Reasonable for local LLMs (Llama, Mistral)
    raptor: str = "off"  # Hierarchical indexing disabled for simplicity
    graph: bool = False  # Graph retrieval disabled
    coverage_target: float = 0.7  # Target 70% coverage
    max_context_tokens: int = 4096  # Safe default for most local models


# Retrieval presets for different use cases
RETRIEVAL_PRESETS = {
    "fast": RetrievalDefaults(
        dense_k=3,
        sparse_k=5,
        top_n=3,
        target_tokens=800,
        coverage_target=0.5,
        max_context_tokens=2048,
    ),
    "balanced": RetrievalDefaults(),  # Uses defaults above
    "thorough": RetrievalDefaults(
        dense_k=10,
        sparse_k=20,
        top_n=8,
        target_tokens=3000,
        coverage_target=0.9,
        max_context_tokens=8192,
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
