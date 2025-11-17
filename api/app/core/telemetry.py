"""Telemetry storage module."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional
import json
import uuid


@dataclass
class Trace:
    id: str
    started_at: datetime
    completed_at: datetime
    prompt: str
    answer: str
    metrics: Dict[str, float]


class TelemetryStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path or Path.cwd() / "data" / "traces.json")
        self._lock = RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._traces: List[Trace] = []
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._traces = [self._decode(item) for item in raw]

    def _decode(self, payload: Dict[str, str]) -> Trace:
        return Trace(
            id=payload["id"],
            started_at=datetime.fromisoformat(payload["started_at"]),
            completed_at=datetime.fromisoformat(payload["completed_at"]),
            prompt=payload["prompt"],
            answer=payload["answer"],
            metrics=payload.get("metrics", {}),
        )

    def _persist(self) -> None:
        payload = []
        for trace in self._traces:
            data = asdict(trace)
            data["started_at"] = trace.started_at.isoformat()
            data["completed_at"] = trace.completed_at.isoformat()
            payload.append(data)
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record(self, prompt: str, answer: str, metrics: Optional[Dict[str, float]] = None) -> Trace:
        with self._lock:
            trace = Trace(
                id=str(uuid.uuid4()),
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                prompt=prompt,
                answer=answer,
                metrics=metrics or {},
            )
            self._traces.append(trace)
            self._persist()
            return trace

    def list(self) -> List[Trace]:
        with self._lock:
            return list(self._traces)
