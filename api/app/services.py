"""Service registry and dependencies for FastAPI routes."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from .core import (
    BQConfig,
    BQHybridRetrievalEngine,
    # v2 Binary Quantization
    BQRetrievalConfig,
    ChunkingStrategy,
    ConfigStore,
    ConversationMemory,
    DocumentStore,
    Gatherer,
    HybridConfig,
    IngestPipeline,
    LocalFirstRegistry,
    MilvusConfig,
    Orchestrator,
    Planner,
    ProviderFactory,
    RetrievalModeV2,
    SmartPlanner,  # Phase 2: Smart planning
    TelemetryStore,
)
from .schemas.config import AppConfig


def _build_retrieval_config(cfg: AppConfig) -> HybridConfig:
    return HybridConfig(
        embedding_model=cfg.retrieval.embedding_model,
        reranker_model=cfg.retrieval.reranker_model,
        use_reranking=cfg.retrieval.use_reranking,
        rerank_top_k=cfg.retrieval.rerank_pool,
        chunking_strategy=ChunkingStrategy(cfg.retrieval.chunking_strategy),
        chunk_size=cfg.retrieval.chunk_size,
        chunk_overlap=cfg.retrieval.chunk_overlap,
        dense_weight=cfg.retrieval.dense_weight if cfg.retrieval.hybrid else 0.0,
        sparse_weight=cfg.retrieval.sparse_weight if cfg.retrieval.hybrid else 1.0,
        raptor=getattr(cfg.retrieval, "raptor", False),
        graph=getattr(cfg.retrieval, "graph", False),
        use_colbert=getattr(cfg.retrieval, "use_colbert", False),
        colbert_model=getattr(cfg.retrieval, "colbert_model", "sentence-transformers/all-MiniLM-L6-v2"),
        colbert_top_k=getattr(cfg.retrieval, "colbert_top_k", 12),
        recency_weight=getattr(cfg.retrieval, "recency_weight", 0.1),
        recency_half_life_days=getattr(cfg.retrieval, "recency_half_life_days", 90.0),
        title_boost=getattr(cfg.retrieval, "title_boost", 0.6),
        heading_boost=getattr(cfg.retrieval, "heading_boost", 0.4),
        proximity_weight=getattr(cfg.retrieval, "proximity_weight", 0.5),
        diversity=getattr(cfg.retrieval, "diversity", 0.0),
        deployment_profile=cfg.deployment_profile.value,
        backend_map={key: backend.backend_id for key, backend in cfg.backends.items()},
        fallback_map={key: fallback.order for key, fallback in cfg.fallbacks.items()},
    )


class ServiceContainer:
    def __init__(self, base_path: Path | None = None) -> None:
        self.demo_mode = os.environ.get("JR_DEMO_MODE", "").lower() in {"1", "true", "yes"}
        self._demo_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        if base_path is None and self.demo_mode and not os.environ.get("JR_DATA_DIR"):
            self._demo_tmpdir = tempfile.TemporaryDirectory(prefix="jr-autorag-demo-")
            data_dir = Path(self._demo_tmpdir.name)
        else:
            data_dir = Path(base_path or os.environ.get("JR_DATA_DIR", Path.cwd() / "data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        self.config_store = ConfigStore(data_dir / "config.json")
        self.document_store = DocumentStore(
            data_dir / "documents.db",
            legacy_json_path=data_dir / "documents.json",
        )
        self.telemetry = TelemetryStore(data_dir / "traces.json")

        # Load config for retrieval settings
        cfg = self.config_store.read()
        sanitized = self._sanitize_config(cfg)
        if sanitized.model_dump() != cfg.model_dump():
            self.config_store.write(sanitized)
            cfg = sanitized
        self.local_first = LocalFirstRegistry(cfg)
        self.memory = ConversationMemory()
        self._enforce_runtime_policy(cfg)

        from .core.auth import get_auth
        from .core.document_acl import get_acl_enforcer, resolve_acl_defaults
        auth_enabled = get_auth().require_auth()
        default_public, _ = resolve_acl_defaults(auth_enabled)
        get_acl_enforcer(default_public=default_public)

        # Configure hybrid retrieval from app config
        retrieval_config = _build_retrieval_config(cfg)

        bq_config, bq_enabled = self._build_bq_config(cfg)
        self.retrieval_engine = BQHybridRetrievalEngine(
            self.document_store,
            retrieval_config,
            bq_config=bq_config,
            bq_enabled=bq_enabled,
        )

        # Try loading cached index first - only rebuild if invalid/stale
        if not self.retrieval_engine.load_index():
            print("HybridRetrievalEngine: No valid cached index, building fresh...")
            self.retrieval_engine.build()
        else:
            print("HybridRetrievalEngine: Loaded cached index successfully!")

        self.provider_factory = ProviderFactory()
        self.ingest = IngestPipeline(
            self.document_store,
            self.retrieval_engine,
            config_getter=self.config_store.read,
            data_dir=data_dir,
            policy_registry=self.local_first,
        )
        self.gatherer = Gatherer(self.retrieval_engine)
        self.simple_planner = Planner(cfg)
        self.smart_planner = SmartPlanner(cfg)
        self.planner = self.smart_planner if cfg.retrieval.planner_mode != "simple" else self.simple_planner
        self.orchestrator = Orchestrator(
            planner=self.planner,
            retrieval=self.retrieval_engine,
            gatherer=self.gatherer,
            provider_factory=self.provider_factory,
            telemetry=self.telemetry,
            memory_store=self.memory,
            policy_registry=self.local_first,
        )
        self.orchestrator.rebuild(cfg)

        # Register orchestrator in global state for traces.py access
        from .state import set_orchestrator
        set_orchestrator(self.orchestrator)

    def _sanitize_config(
        self,
        cfg: AppConfig,
        existing: AppConfig | None = None,
    ) -> AppConfig:
        from .core.providers import _infer_secret_key_name
        from .core.secrets_vault import get_secrets_vault

        vault = get_secrets_vault()

        def store_secret(key_name: str, value: str) -> bool:
            try:
                vault.set(key_name, value)
                return True
            except Exception:
                return False

        def sanitize_provider(provider, existing_provider):
            if provider is None:
                return None
            key_name = _infer_secret_key_name(provider.name, str(provider.base_url))
            candidate = (provider.api_key or "").strip()
            existing_key_name = (
                _infer_secret_key_name(existing_provider.name, str(existing_provider.base_url))
                if existing_provider
                else None
            )
            fallback = (
                (existing_provider.api_key or "").strip()
                if existing_provider and existing_key_name == key_name
                else ""
            )

            if candidate:
                if store_secret(key_name, candidate):
                    return provider.model_copy(update={"api_key": None})
                return provider

            if fallback:
                if store_secret(key_name, fallback):
                    return provider.model_copy(update={"api_key": None})
                return provider.model_copy(update={"api_key": fallback})

            if vault.get(key_name):
                return provider.model_copy(update={"api_key": None})

            return provider.model_copy(update={"api_key": None})

        sanitized_provider = sanitize_provider(cfg.provider, existing.provider if existing else None)
        existing_profiles = {p.name: p for p in (existing.provider_profiles if existing else [])}
        sanitized_profiles = []
        for profile in cfg.provider_profiles:
            existing_profile = existing_profiles.get(profile.name)
            provider = sanitize_provider(
                profile.provider,
                existing_profile.provider if existing_profile else None,
            )
            sanitized_profiles.append(profile.model_copy(update={"provider": provider}))

        return cfg.model_copy(
            update={"provider": sanitized_provider, "provider_profiles": sanitized_profiles}
        )

    def prepare_config_for_storage(self, cfg: AppConfig) -> AppConfig:
        """Store any secrets in vault and redact them from config."""
        current = self.config_store.read()
        return self._sanitize_config(cfg, existing=current)

    def _build_bq_config(self, cfg: AppConfig) -> tuple[BQRetrievalConfig, bool]:
        """Build BQ retrieval configuration from app config."""
        # Get embedding dimension from model info
        from .core.hybrid_retrieval import EmbeddingModelPreset
        model_info = EmbeddingModelPreset.get_info(cfg.retrieval.embedding_model)
        embedding_dim = model_info.get("dimensions", 768)

        milvus_config = MilvusConfig(
            host=getattr(cfg.retrieval, "milvus_host", "localhost"),
            port=getattr(cfg.retrieval, "milvus_port", 19530),
            collection_name=getattr(cfg.retrieval, "milvus_collection", "jr_autorag_chunks_bq"),
            index_type=getattr(cfg.retrieval, "milvus_index_type", "BIN_FLAT"),
            metric_type=getattr(cfg.retrieval, "milvus_metric", "HAMMING"),
            nlist=getattr(cfg.retrieval, "milvus_nlist", 128),
            nprobe=getattr(cfg.retrieval, "milvus_nprobe", 16),
        )

        bq_config = BQConfig(
            rule=getattr(cfg.retrieval, "bq_rule", "sign_threshold_0"),
            normalize=getattr(cfg.retrieval, "bq_normalize", False),
        )

        retrieval_mode = RetrievalModeV2.from_string(
            getattr(cfg.retrieval, "retrieval_mode", "float32")
        )
        bq_enabled = bool(getattr(cfg.retrieval, "bq_enabled", False)) or (
            retrieval_mode == RetrievalModeV2.BINARY
        )

        bq_retrieval_config = BQRetrievalConfig(
            default_mode=retrieval_mode,
            top_k=cfg.retrieval.top_n,
            two_stage_enabled=getattr(cfg.retrieval, "bq_two_stage", False),
            stage1_candidates=getattr(cfg.retrieval, "bq_stage1_candidates", 50),
            fallback_enabled=getattr(cfg.retrieval, "bq_fallback_enabled", True),
            fallback_distance_threshold=getattr(cfg.retrieval, "bq_fallback_threshold", 500.0),
            milvus_config=milvus_config,
            bq_config=bq_config,
            embedding_dim=embedding_dim,
            embedding_model=cfg.retrieval.embedding_model,
        )
        return bq_retrieval_config, bq_enabled

    def _enforce_runtime_policy(self, cfg: AppConfig) -> None:
        subsystems = [
            "document_parser",
            "ocr",
            "embedding",
            "reranker",
            "vector_store",
            "sparse_index",
            "graph_store",
            "memory",
            "eval",
            "telemetry",
        ]
        for subsystem in subsystems:
            self.local_first.ensure_runtime_allowed(subsystem)

    def apply_config(self, cfg: AppConfig) -> None:
        self.local_first.refresh(cfg)
        self._enforce_runtime_policy(cfg)
        self.simple_planner.rebuild(cfg)
        self.smart_planner.rebuild(cfg)
        self.planner = self.smart_planner if cfg.retrieval.planner_mode != "simple" else self.simple_planner
        self.orchestrator.set_planner(self.planner)
        retrieval_config = _build_retrieval_config(cfg)
        self.retrieval_engine.reconfigure(retrieval_config)
        self.orchestrator.rebuild(cfg)

        if hasattr(self.retrieval_engine, "set_bq_config"):
            bq_config, bq_enabled = self._build_bq_config(cfg)
            self.retrieval_engine.set_bq_config(bq_config, enabled=bq_enabled, rebuild=True)


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    return ServiceContainer()
