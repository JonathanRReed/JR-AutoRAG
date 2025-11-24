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
    Planner,
    ProviderFactory,
    RetrievalEngine,
    TelemetryStore,
)


class ServiceContainer:
    def __init__(self, base_path: Path | None = None) -> None:
        data_dir = Path(base_path or os.environ.get("JR_DATA_DIR", Path.cwd() / "data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        self.config_store = ConfigStore(data_dir / "config.json")
        self.document_store = DocumentStore(data_dir / "documents.json")
        self.telemetry = TelemetryStore(data_dir / "traces.json")
        self.retrieval_engine = RetrievalEngine(self.document_store)
        self.retrieval_engine.build()
        self.ingest = IngestPipeline(self.document_store, self.retrieval_engine)
        self.gatherer = Gatherer(self.retrieval_engine)
        cfg = self.config_store.read()
        self.planner = Planner(cfg)
        self.provider_factory = ProviderFactory()
        self.orchestrator = Orchestrator(
            planner=self.planner,
            retrieval=self.retrieval_engine,
            gatherer=self.gatherer,
            provider_factory=self.provider_factory,
            telemetry=self.telemetry,
        )
        self.orchestrator.rebuild(cfg)


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    return ServiceContainer()
