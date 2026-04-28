"""Local AutoRAG-style experiment persistence and ranking."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock

from ..schemas.config import AppConfig, RETRIEVAL_PRESETS


@dataclass
class EvalMetricResult:
    name: str
    value: float
    provider: str = "local_heuristic"
    direction: str = "higher_is_better"
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    parser: list[str] = field(default_factory=lambda: ["native", "docling"])
    chunker: list[str] = field(default_factory=lambda: ["recursive"])
    embedding: list[str] = field(default_factory=list)
    dense_weight: list[float] = field(default_factory=lambda: [0.55, 0.65, 0.75])
    sparse_weight: list[float] = field(default_factory=lambda: [0.45, 0.35, 0.25])
    reranker: list[bool] = field(default_factory=lambda: [True, False])
    graph: list[bool] = field(default_factory=lambda: [False, True])
    raptor: list[bool] = field(default_factory=lambda: [False, True])
    ocr_policy: list[str] = field(default_factory=lambda: ["auto"])
    questions: list[str] = field(default_factory=list)


@dataclass
class ExperimentRun:
    id: str
    config: ExperimentConfig
    status: str
    created_at: str
    owner_id: str = "anonymous"
    completed_at: str | None = None
    metrics: list[EvalMetricResult] = field(default_factory=list)
    winning_preset: str | None = None
    config_snapshot: dict[str, object] = field(default_factory=dict)
    traces: list[str] = field(default_factory=list)
    promoted_at: str | None = None


class ExperimentRunStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path or Path.cwd() / "data" / "experiments.json")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _read(self) -> list[dict[str, object]]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def _write(self, runs: list[dict[str, object]]) -> None:
        self._path.write_text(json.dumps(runs, indent=2, sort_keys=True), encoding="utf-8")

    def list(self, limit: int = 50, owner_id: str | None = None) -> list[dict[str, object]]:
        with self._lock:
            runs = self._read()
            if owner_id is not None:
                runs = [run for run in runs if run.get("owner_id", "anonymous") == owner_id]
            return list(reversed(runs))[:limit]

    def get(self, run_id: str, owner_id: str | None = None) -> dict[str, object] | None:
        with self._lock:
            for run in self._read():
                if run.get("id") == run_id:
                    if owner_id is not None and run.get("owner_id", "anonymous") != owner_id:
                        return None
                    return run
        return None

    def save(self, run: ExperimentRun) -> dict[str, object]:
        payload = asdict(run)
        with self._lock:
            runs = [item for item in self._read() if item.get("id") != run.id]
            runs.append(payload)
            self._write(runs)
        return payload

    def mark_promoted(self, run_id: str) -> dict[str, object] | None:
        with self._lock:
            runs = self._read()
            target: dict[str, object] | None = None
            for run in runs:
                if run.get("id") == run_id:
                    run["promoted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    target = run
                    break
            if target is not None:
                self._write(runs)
            return target


class LocalExperimentRunner:
    def __init__(self, store: ExperimentRunStore) -> None:
        self._store = store

    def run(
        self,
        cfg: ExperimentConfig,
        app_config: AppConfig,
        doc_count: int,
        owner_id: str = "anonymous",
    ) -> dict[str, object]:
        retrieval = app_config.retrieval
        dense_weight = float(getattr(retrieval, "dense_weight", 0.65) or 0.65)
        sparse_weight = float(getattr(retrieval, "sparse_weight", 0.35) or 0.35)
        feature_score = sum(
            [
                0.15 if getattr(retrieval, "use_reranking", False) else 0.0,
                0.12 if getattr(retrieval, "graph", False) else 0.0,
                0.10 if getattr(retrieval, "raptor", False) else 0.0,
                0.08 if getattr(retrieval, "enforce_evidence_contract", False) else 0.0,
            ]
        )
        balance_score = max(0.0, 1.0 - abs((dense_weight + sparse_weight) - 1.0))
        corpus_score = min(doc_count / 12.0, 1.0)
        matrix_score = sum(
            [
                0.03 if "docling" in {parser.lower() for parser in cfg.parser} else 0.0,
                0.02 if any(cfg.reranker) else 0.0,
                0.02 if any(cfg.graph) else 0.0,
                0.02 if any(cfg.raptor) else 0.0,
                0.02 if cfg.questions else 0.0,
            ]
        )
        readiness = min(
            1.0,
            0.45 + feature_score + (balance_score * 0.25) + (corpus_score * 0.15) + matrix_score,
        )
        winning_preset = self._recommend_preset(app_config, readiness)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        run_id = str(uuid.uuid4())
        run = ExperimentRun(
            id=run_id,
            config=cfg,
            status="completed",
            created_at=now,
            owner_id=owner_id,
            completed_at=now,
            metrics=[
                EvalMetricResult("faithfulness", readiness),
                EvalMetricResult("context_precision", min(1.0, readiness + 0.06)),
                EvalMetricResult("context_recall", min(1.0, readiness + (0.04 if doc_count else -0.2))),
                EvalMetricResult("local_only_compliance", 1.0 if app_config.deployment_profile.value == "local_only" else 0.75),
                EvalMetricResult("matrix_coverage", min(1.0, matrix_score / 0.11)),
            ],
            winning_preset=winning_preset,
            config_snapshot=app_config.model_dump(mode="json"),
            traces=[
                f"experiment:{run_id}:parser={','.join(cfg.parser)}",
                f"experiment:{run_id}:weights={dense_weight:.2f}/{sparse_weight:.2f}",
                f"experiment:{run_id}:preset={winning_preset}",
            ],
        )
        return self._store.save(run)

    def _recommend_preset(self, app_config: AppConfig, readiness: float) -> str:
        if readiness >= 0.82:
            return "ultra_accurate"
        if getattr(app_config.retrieval, "use_reranking", False):
            return "thorough"
        if app_config.deployment_profile.value == "local_only":
            return "balanced"
        return "fast"


def apply_winning_preset(app_config: AppConfig, preset_name: str | None) -> AppConfig:
    preset = RETRIEVAL_PRESETS.get((preset_name or "balanced").lower())
    if preset is None:
        preset = RETRIEVAL_PRESETS["balanced"]
    return app_config.model_copy(update={"retrieval": preset.model_copy()})
