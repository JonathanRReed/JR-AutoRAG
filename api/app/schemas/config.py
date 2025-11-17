from enum import Enum
from typing import List, Optional

from pydantic import AnyHttpUrl, BaseModel


class ProviderConfig(BaseModel):
    name: str
    base_url: AnyHttpUrl
    planner_model: Optional[str] = None
    gatherer_model: Optional[str] = None
    generator_model: Optional[str] = None
    api_key: Optional[str] = None


class ProviderProfile(BaseModel):
    name: str
    provider: ProviderConfig


class RetrievalDefaults(BaseModel):
    hybrid: bool = True
    dense_k: int = 40
    sparse_k: int = 80
    rerank_pool: int = 50
    top_n: int = 12
    compression: bool = True
    target_tokens: int = 1600
    raptor: str = "hierarchical"  # off|simple|hierarchical
    graph: bool = False
    coverage_target: float = 0.7
    max_context_tokens: int = 4096


class AppConfig(BaseModel):
    profile: str = "Default"
    provider: Optional[ProviderConfig] = None
    provider_profiles: List[ProviderProfile] = []
    retrieval: RetrievalDefaults = RetrievalDefaults()


class ProviderKind(str, Enum):
    OLLAMA = "ollama"
    LM_STUDIO = "lmstudio"
    OPENAI_COMPAT = "openai"


class LocalProviderInfo(BaseModel):
    kind: ProviderKind
    name: str
    base_url: AnyHttpUrl
    models: List[str] = []
    running: List[str] = []
    version: Optional[str] = None
    status: str = "ok"
    error_message: Optional[str] = None
