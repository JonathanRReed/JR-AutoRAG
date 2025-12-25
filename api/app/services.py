"""Service registry and dependencies for FastAPI routes."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .core import (
    ConfigStore,
    DocumentStore,
    Gatherer,
    IngestPipeline,
    Orchestrator,
    ProviderFactory,
    TelemetryStore,
    HybridConfig,
    ChunkingStrategy,
    BQHybridRetrievalEngine,
    Planner,
    SmartPlanner,  # Phase 2: Smart planning
    # v2 Binary Quantization
    BQRetrievalConfig,
    MilvusConfig,
    BQConfig,
    RetrievalModeV2,
)

def _build_retrieval_config(cfg: "AppConfig") -> HybridConfig:
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
    )


class ServiceContainer:
    def __init__(self, base_path: Path | None = None) -> None:
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
        
        self.ingest = IngestPipeline(self.document_store, self.retrieval_engine)
        self.gatherer = Gatherer(self.retrieval_engine)
        self.simple_planner = Planner(cfg)
        self.smart_planner = SmartPlanner(cfg)
        self.planner = self.smart_planner if cfg.retrieval.planner_mode != "simple" else self.simple_planner
        self.provider_factory = ProviderFactory()
        self.orchestrator = Orchestrator(
            planner=self.planner,
            retrieval=self.retrieval_engine,
            gatherer=self.gatherer,
            provider_factory=self.provider_factory,
            telemetry=self.telemetry,
        )
        self.orchestrator.rebuild(cfg)
        
        # Register orchestrator in global state for traces.py access
        from .state import set_orchestrator
        set_orchestrator(self.orchestrator)

    def _build_bq_config(self, cfg: "AppConfig") -> tuple[BQRetrievalConfig, bool]:
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

    def apply_config(self, cfg: AppConfig) -> None:
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
