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
    HybridRetrievalEngine,
    HybridConfig,
    ChunkingStrategy,
    Planner,
    SmartPlanner,  # Phase 2: Smart planning
)


class ServiceContainer:
    def __init__(self, base_path: Path | None = None) -> None:
        data_dir = Path(base_path or os.environ.get("JR_DATA_DIR", Path.cwd() / "data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        self.config_store = ConfigStore(data_dir / "config.json")
        self.document_store = DocumentStore(data_dir / "documents.json")
        self.telemetry = TelemetryStore(data_dir / "traces.json")
        
        # Load config for retrieval settings
        cfg = self.config_store.read()
        
        # Configure hybrid retrieval from app config
        retrieval_config = HybridConfig(
            embedding_model=cfg.retrieval.embedding_model,
            reranker_model=cfg.retrieval.reranker_model,
            use_reranking=cfg.retrieval.use_reranking,
            rerank_top_k=cfg.retrieval.rerank_pool,
            chunking_strategy=ChunkingStrategy(cfg.retrieval.chunking_strategy),
            chunk_size=cfg.retrieval.chunk_size,
            chunk_overlap=cfg.retrieval.chunk_overlap,
            dense_weight=cfg.retrieval.dense_weight if cfg.retrieval.hybrid else 0.0,
            sparse_weight=cfg.retrieval.sparse_weight if cfg.retrieval.hybrid else 1.0,
            raptor=getattr(cfg.retrieval, 'raptor', False),
            graph=getattr(cfg.retrieval, 'graph', False),
            recency_weight=getattr(cfg.retrieval, "recency_weight", 0.1),
            recency_half_life_days=getattr(cfg.retrieval, "recency_half_life_days", 90.0),
            title_boost=getattr(cfg.retrieval, "title_boost", 0.6),
            heading_boost=getattr(cfg.retrieval, "heading_boost", 0.4),
            proximity_weight=getattr(cfg.retrieval, "proximity_weight", 0.5),
            diversity=getattr(cfg.retrieval, "diversity", 0.0),
        )
        
        self.retrieval_engine = HybridRetrievalEngine(self.document_store, retrieval_config)
        self.retrieval_engine.build()
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

    def apply_config(self, cfg: AppConfig) -> None:
        self.simple_planner.rebuild(cfg)
        self.smart_planner.rebuild(cfg)
        self.planner = self.smart_planner if cfg.retrieval.planner_mode != "simple" else self.simple_planner
        self.orchestrator.set_planner(self.planner)
        self.orchestrator.rebuild(cfg)


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    return ServiceContainer()
