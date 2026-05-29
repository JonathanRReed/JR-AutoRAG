from __future__ import annotations

from enum import Enum
from ipaddress import ip_address
import socket
from typing import Literal
from urllib.parse import urlparse

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator, model_validator


class DeploymentProfile(str, Enum):
    LOCAL_ONLY = "local_only"
    CLIENT_SAFE = "client_safe"
    HYBRID = "hybrid"
    CLOUD_ACCELERATED = "cloud_accelerated"


class BackendMode(str, Enum):
    LOCAL = "local"
    HYBRID = "hybrid"
    CLOUD = "cloud"


class CapabilityClass(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SubsystemType(str, Enum):
    DOCUMENT_PARSER = "document_parser"
    OCR = "ocr"
    EMBEDDING = "embedding"
    RERANKER = "reranker"
    VECTOR_STORE = "vector_store"
    SPARSE_INDEX = "sparse_index"
    GRAPH_STORE = "graph_store"
    LLM = "llm"
    MEMORY = "memory"
    EVAL = "eval"
    TELEMETRY = "telemetry"


class OCRPolicy(str, Enum):
    OFF = "off"
    AUTO = "auto"
    VISION_MODEL = "vision_model"
    DEDICATED_OCR = "dedicated_ocr"
    HYBRID = "hybrid"


def _is_local_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        parsed_ip = ip_address(host)
    except ValueError:
        return False
    return parsed_ip.is_loopback or parsed_ip.is_unspecified


def _is_client_owned_url(value: str) -> bool:
    parsed = urlparse(value)
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host in {"localhost"} or host.endswith(".localhost"):
        return True
    try:
        parsed_ip = ip_address(host)
    except ValueError:
        if not host.endswith((".local", ".lan", ".internal")):
            return False
        try:
            resolved = {
                ip_address(result[4][0])
                for result in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            }
        except OSError:
            return False
        return bool(resolved) and all(
            item.is_loopback or item.is_private or item.is_link_local for item in resolved
        )
    return parsed_ip.is_loopback or parsed_ip.is_private or parsed_ip.is_link_local


class BackendCapabilities(BaseModel):
    mode: BackendMode = BackendMode.LOCAL
    requires_network: bool = False
    supports_batching: bool = False
    supports_streaming: bool = False
    supports_multimodal: bool = False
    estimated_latency_class: CapabilityClass = CapabilityClass.MEDIUM
    estimated_memory_class: CapabilityClass = CapabilityClass.MEDIUM


class BackendConfig(BaseModel):
    subsystem: SubsystemType
    backend_id: str
    label: str
    enabled: bool = True
    capabilities: BackendCapabilities = Field(default_factory=BackendCapabilities)
    settings: dict[str, str | int | float | bool] = Field(default_factory=dict)


class FallbackConfig(BaseModel):
    enabled: bool = True
    order: list[str] = Field(default_factory=list)
    on_failure: Literal["error", "fallback"] = "fallback"


class OCRSettings(BaseModel):
    policy: OCRPolicy = OCRPolicy.AUTO
    extractable_text_threshold: float = 0.65
    min_characters: int = 80
    allow_cloud_fallback: bool = False
    preferred_backends: list[str] = Field(
        default_factory=lambda: [
            "ocr.local.tesseract",
            "ocr.local.vision",
        ]
    )
    dual_merge_strategy: Literal["highest_confidence", "prefer_text_parser"] = "highest_confidence"

    @field_validator("extractable_text_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        return min(max(value, 0.0), 1.0)


class IngestSettings(BaseModel):
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    parsing_stack: list[str] = Field(
        default_factory=lambda: [
            "native_text",
            "layout_parser",
            "ocr",
            "vision_rescue",
        ]
    )
    attach_processing_trace: bool = True


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


def build_default_backends() -> dict[str, BackendConfig]:
    return {
        "document_parser": BackendConfig(
            subsystem=SubsystemType.DOCUMENT_PARSER,
            backend_id="document_parser.local.native",
            label="Native Parser Stack",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                supports_batching=True,
                estimated_latency_class=CapabilityClass.LOW,
                estimated_memory_class=CapabilityClass.LOW,
            ),
        ),
        "ocr": BackendConfig(
            subsystem=SubsystemType.OCR,
            backend_id="ocr.local.tesseract",
            label="Tesseract OCR",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                supports_batching=True,
                supports_multimodal=True,
                estimated_latency_class=CapabilityClass.MEDIUM,
                estimated_memory_class=CapabilityClass.LOW,
            ),
            settings={
                "vision_model": "",
                "max_pages": 8,
            },
        ),
        "embedding": BackendConfig(
            subsystem=SubsystemType.EMBEDDING,
            backend_id="embedding.local.sentence_transformer",
            label="Local Sentence Transformer",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                supports_batching=True,
                estimated_latency_class=CapabilityClass.MEDIUM,
                estimated_memory_class=CapabilityClass.MEDIUM,
            ),
        ),
        "reranker": BackendConfig(
            subsystem=SubsystemType.RERANKER,
            backend_id="reranker.local.cross_encoder",
            label="Local Cross Encoder",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                supports_batching=True,
                estimated_latency_class=CapabilityClass.MEDIUM,
                estimated_memory_class=CapabilityClass.MEDIUM,
            ),
        ),
        "vector_store": BackendConfig(
            subsystem=SubsystemType.VECTOR_STORE,
            backend_id="vector_store.local.hybrid",
            label="Local Hybrid Vector Store",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                supports_batching=True,
                estimated_latency_class=CapabilityClass.MEDIUM,
                estimated_memory_class=CapabilityClass.HIGH,
            ),
        ),
        "sparse_index": BackendConfig(
            subsystem=SubsystemType.SPARSE_INDEX,
            backend_id="sparse_index.local.bm25",
            label="Local BM25",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                supports_batching=True,
                estimated_latency_class=CapabilityClass.LOW,
                estimated_memory_class=CapabilityClass.LOW,
            ),
        ),
        "graph_store": BackendConfig(
            subsystem=SubsystemType.GRAPH_STORE,
            backend_id="graph_store.local.in_process",
            label="In-Process GraphRAG",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                estimated_latency_class=CapabilityClass.MEDIUM,
                estimated_memory_class=CapabilityClass.MEDIUM,
            ),
        ),
        "llm": BackendConfig(
            subsystem=SubsystemType.LLM,
            backend_id="llm.local.provider",
            label="Configured Local Provider",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                supports_batching=True,
                supports_streaming=True,
                supports_multimodal=True,
                estimated_latency_class=CapabilityClass.MEDIUM,
                estimated_memory_class=CapabilityClass.HIGH,
            ),
        ),
        "memory": BackendConfig(
            subsystem=SubsystemType.MEMORY,
            backend_id="memory.local.session",
            label="Local Memory Store",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                estimated_latency_class=CapabilityClass.LOW,
                estimated_memory_class=CapabilityClass.LOW,
            ),
        ),
        "eval": BackendConfig(
            subsystem=SubsystemType.EVAL,
            backend_id="eval.local.harness",
            label="Local Evaluation Harness",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                supports_batching=True,
                estimated_latency_class=CapabilityClass.MEDIUM,
                estimated_memory_class=CapabilityClass.MEDIUM,
            ),
        ),
        "telemetry": BackendConfig(
            subsystem=SubsystemType.TELEMETRY,
            backend_id="telemetry.local.json",
            label="Local JSON Telemetry",
            capabilities=BackendCapabilities(
                mode=BackendMode.LOCAL,
                estimated_latency_class=CapabilityClass.LOW,
                estimated_memory_class=CapabilityClass.LOW,
            ),
        ),
    }


def build_default_fallbacks() -> dict[str, FallbackConfig]:
    return {
        "document_parser": FallbackConfig(order=["document_parser.local.native"]),
        "ocr": FallbackConfig(order=["ocr.local.tesseract", "ocr.local.vision"]),
        "embedding": FallbackConfig(order=["embedding.local.sentence_transformer"]),
        "reranker": FallbackConfig(order=["reranker.local.cross_encoder"]),
        "vector_store": FallbackConfig(order=["vector_store.local.hybrid"]),
        "sparse_index": FallbackConfig(order=["sparse_index.local.bm25"]),
        "graph_store": FallbackConfig(order=["graph_store.local.in_process"]),
        "llm": FallbackConfig(order=["llm.local.provider"]),
        "memory": FallbackConfig(order=["memory.local.session"]),
        "eval": FallbackConfig(order=["eval.local.harness"]),
        "telemetry": FallbackConfig(order=["telemetry.local.json"]),
    }


def build_known_backend_ids() -> set[str]:
    known = {backend.backend_id for backend in build_default_backends().values()}
    known.update(
        {
            "ocr.local.vision",
        }
    )
    return known


class RetrievalDefaults(BaseModel):
    """Retrieval configuration optimized for hybrid retrieval."""

    hybrid: bool = True
    dense_k: int = 5
    sparse_k: int = 10
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rerank_pool: int = 20
    top_n: int = 5
    compression: bool = False
    target_tokens: int = 1600
    raptor: bool = False
    graph: bool = False
    coverage_target: float = 0.7
    max_context_tokens: int = 4096
    chunking_strategy: str = "semantic"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranking: bool = True
    use_colbert: bool = False
    colbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    colbert_top_k: int = 12
    chunk_size: int = 400
    chunk_overlap: int = 50
    planner_mode: str = "smart"
    flare_generation: bool = True
    enforce_evidence_contract: bool = True
    multi_resolution: bool = True
    recency_weight: float = 0.1
    recency_half_life_days: float = 90.0
    title_boost: float = 0.6
    heading_boost: float = 0.4
    proximity_weight: float = 0.5
    diversity: float = 0.0
    use_hyde: bool = False
    abstain_when_unverified: bool = False
    self_rag_critic: bool = False
    retrieval_mode: str = "float32"
    bq_enabled: bool = False
    bq_normalize: bool = False
    bq_rule: str = "sign_threshold_0"
    bq_two_stage: bool = False
    bq_stage1_candidates: int = 50
    bq_fallback_enabled: bool = True
    bq_fallback_threshold: float = 500.0
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "jr_autorag_chunks_bq"
    milvus_index_type: str = "BIN_FLAT"
    milvus_metric: str = "HAMMING"
    milvus_nlist: int = 128
    milvus_nprobe: int = 16
    langextract_enabled: bool = False
    langextract_profile_default: str = "generic_entities_v1"
    langextract_model_source: Literal["planner", "gatherer", "generator"] = "gatherer"
    langextract_timeout_sec: int = 20
    langextract_max_chars: int = 12000
    langextract_max_synthetic_facts: int = 200

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
        use_reranking=False,
        chunking_strategy="fixed",
        chunk_size=800,
        flare_generation=False,
        enforce_evidence_contract=False,
        multi_resolution=False,
        raptor=False,
        graph=False,
        retrieval_mode="binary",
        bq_enabled=True,
        bq_normalize=False,
        bq_rule="sign_threshold_0",
        bq_two_stage=False,
        bq_stage1_candidates=20,
        bq_fallback_enabled=False,
        bq_fallback_threshold=700.0,
        milvus_index_type="BIN_FLAT",
        milvus_metric="HAMMING",
        milvus_nlist=64,
        milvus_nprobe=8,
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
        retrieval_mode="binary",
        bq_enabled=True,
        bq_normalize=False,
        bq_rule="sign_threshold_0",
        bq_two_stage=False,
        bq_stage1_candidates=50,
        bq_fallback_enabled=True,
        bq_fallback_threshold=600.0,
        milvus_index_type="BIN_FLAT",
        milvus_metric="HAMMING",
        milvus_nlist=128,
        milvus_nprobe=12,
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
        abstain_when_unverified=False,
        retrieval_mode="binary",
        bq_enabled=True,
        bq_normalize=True,
        bq_rule="sign_threshold_0",
        bq_two_stage=True,
        bq_stage1_candidates=100,
        bq_fallback_enabled=True,
        bq_fallback_threshold=500.0,
        milvus_index_type="BIN_FLAT",
        milvus_metric="HAMMING",
        milvus_nlist=128,
        milvus_nprobe=16,
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
        abstain_when_unverified=False,
        self_rag_critic=False,
        retrieval_mode="binary",
        bq_enabled=True,
        bq_normalize=True,
        bq_rule="sign_threshold_0",
        bq_two_stage=True,
        bq_stage1_candidates=150,
        bq_fallback_enabled=True,
        bq_fallback_threshold=400.0,
        milvus_index_type="BIN_FLAT",
        milvus_metric="HAMMING",
        milvus_nlist=128,
        milvus_nprobe=24,
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
        abstain_when_unverified=False,
        self_rag_critic=False,
        retrieval_mode="float32",
        bq_enabled=False,
        bq_normalize=True,
        bq_rule="sign_threshold_0",
        bq_two_stage=False,
        bq_stage1_candidates=200,
        bq_fallback_enabled=True,
        bq_fallback_threshold=400.0,
        milvus_index_type="BIN_FLAT",
        milvus_metric="HAMMING",
        milvus_nlist=128,
        milvus_nprobe=32,
    ),
}


class StageBudgetDefaults(BaseModel):
    planner_timeout_ms: int = 3000
    gatherer_timeout_ms: int = 12000
    rerank_timeout_ms: int = 5000
    compression_timeout_ms: int = 4000
    generation_timeout_ms: int = 20000
    verification_timeout_ms: int = 5000
    total_timeout_ms: int = 60000
    retrieval_token_budget: int = 8000
    rerank_pool_budget: int = 50
    compression_token_budget: int = 4000
    answer_token_budget: int = 2000


class ClientDataPolicy(BaseModel):
    classification: Literal["internal", "client_confidential", "regulated"] = "client_confidential"
    storage_boundary: Literal["local_only", "client_owned"] = "client_owned"
    managed_cloud_hosting_allowed: bool = False
    external_model_calls_allowed: bool = False
    pii_redaction_required: bool = True
    document_retention_days: int = 30
    trace_retention_days: int = 14
    report_export_mode: Literal["redacted_by_default", "full_with_client_approval"] = "redacted_by_default"
    client_handoff_required: bool = True
    operator_review_required: bool = True

    @field_validator("document_retention_days", "trace_retention_days")
    @classmethod
    def _validate_retention(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Retention days must be zero or greater.")
        return value


class AppConfig(BaseModel):
    profile: str = "Default"
    deployment_profile: DeploymentProfile = DeploymentProfile.LOCAL_ONLY
    data_policy: ClientDataPolicy = Field(default_factory=ClientDataPolicy)
    provider: ProviderConfig | None = None
    provider_profiles: list[ProviderProfile] = Field(default_factory=list)
    retrieval: RetrievalDefaults = Field(default_factory=RetrievalDefaults)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    backends: dict[str, BackendConfig] = Field(default_factory=build_default_backends)
    fallbacks: dict[str, FallbackConfig] = Field(default_factory=build_default_fallbacks)
    query_mode: str = "grounded"
    stage_budgets: StageBudgetDefaults = Field(default_factory=StageBudgetDefaults)

    @field_validator("backends", mode="before")
    @classmethod
    def _merge_backend_defaults(cls, value):
        defaults = build_default_backends()
        if value is None:
            return defaults
        if not isinstance(value, dict):
            return value

        merged: dict[str, BackendConfig | dict] = {}
        for key, default in defaults.items():
            raw = value.get(key)
            if raw is None:
                merged[key] = default
            elif isinstance(raw, BackendConfig):
                merged[key] = raw
            elif isinstance(raw, dict):
                merged[key] = {
                    "subsystem": raw.get("subsystem", default.subsystem.value),
                    "backend_id": raw.get("backend_id", default.backend_id),
                    "label": raw.get("label", default.label),
                    "enabled": raw.get("enabled", default.enabled),
                    "capabilities": raw.get("capabilities", default.capabilities.model_dump()),
                    "settings": raw.get("settings", default.settings),
                }
            else:
                merged[key] = default

        for key, raw in value.items():
            if key not in merged:
                merged[key] = raw
        return merged

    @field_validator("fallbacks", mode="before")
    @classmethod
    def _merge_fallback_defaults(cls, value):
        defaults = build_default_fallbacks()
        if value is None:
            return defaults
        if not isinstance(value, dict):
            return value

        merged: dict[str, FallbackConfig | dict] = {}
        for key, default in defaults.items():
            raw = value.get(key)
            if raw is None:
                merged[key] = default
            elif isinstance(raw, FallbackConfig):
                merged[key] = raw
            elif isinstance(raw, dict):
                merged[key] = {
                    "enabled": raw.get("enabled", default.enabled),
                    "order": raw.get("order", default.order),
                    "on_failure": raw.get("on_failure", default.on_failure),
                }
            else:
                merged[key] = default

        for key, raw in value.items():
            if key not in merged:
                merged[key] = raw
        return merged

    @model_validator(mode="after")
    def _validate_local_first_policy(self):
        known_backend_ids = build_known_backend_ids()
        known_backend_ids.update(backend.backend_id for backend in self.backends.values())
        for subsystem, fallback in self.fallbacks.items():
            unknown = [item for item in fallback.order if item not in known_backend_ids]
            if unknown:
                raise ValueError(
                    f"Fallback chain for '{subsystem}' references unknown backends: {', '.join(unknown)}"
                )

        if self.deployment_profile == DeploymentProfile.LOCAL_ONLY:
            if self.provider and not _is_local_url(str(self.provider.base_url)):
                raise ValueError("Local-only mode requires a localhost or loopback provider URL.")
            for profile in self.provider_profiles:
                if not _is_local_url(str(profile.provider.base_url)):
                    raise ValueError(
                        f"Provider profile '{profile.name}' is not compatible with local-only mode: "
                        "provider URL must be localhost or loopback."
                    )

            for name, backend in self.backends.items():
                if not backend.enabled:
                    continue
                if backend.capabilities.requires_network or backend.capabilities.mode != BackendMode.LOCAL:
                    raise ValueError(
                        f"Backend '{name}' is not compatible with local-only mode: {backend.backend_id}"
                    )

            if self.ingest.ocr.allow_cloud_fallback:
                raise ValueError("Local-only mode cannot enable cloud OCR fallback.")

        if self.deployment_profile == DeploymentProfile.CLIENT_SAFE:
            if self.data_policy.document_retention_days > 90 or self.data_policy.trace_retention_days > 90:
                raise ValueError("Client-safe retention may not exceed 90 days without an exception.")
            if self.data_policy.managed_cloud_hosting_allowed:
                raise ValueError("Client-safe mode cannot allow managed cloud hosting.")
            if self.data_policy.external_model_calls_allowed:
                raise ValueError("Client-safe mode cannot allow external model calls.")
            if self.data_policy.storage_boundary not in {"local_only", "client_owned"}:
                raise ValueError("Client-safe mode requires local or client-owned storage.")
            if self.provider and not _is_client_owned_url(str(self.provider.base_url)):
                raise ValueError(
                    "Client-safe mode requires a localhost, private-network, or client-owned provider URL."
                )
            for name, backend in self.backends.items():
                if not backend.enabled:
                    continue
                if backend.capabilities.mode == BackendMode.CLOUD:
                    raise ValueError(
                        f"Backend '{name}' is not compatible with client-safe mode: {backend.backend_id}"
                    )
            if self.ingest.ocr.allow_cloud_fallback:
                raise ValueError("Client-safe mode cannot enable cloud OCR fallback.")

        return self


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
    models: list[str] = Field(default_factory=list)
    running: list[str] = Field(default_factory=list)
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
    deployment_profile: DeploymentProfile = DeploymentProfile.LOCAL_ONLY
    local_only_ready: bool = True


class ModelDownloadRequest(BaseModel):
    kind: str
    model: str
