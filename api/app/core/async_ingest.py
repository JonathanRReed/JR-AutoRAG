"""Async ingestion job model with progress tracking.

This module provides:
- IngestJob: Background job model for document ingestion
- JobStore: In-memory storage for job tracking
- Progress streaming via callbacks
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ingest import IngestPipeline


class JobStatus(str, Enum):
    """Status of an ingestion job."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress update for a job."""
    current: int
    total: int
    stage: str
    message: str = ""

    @property
    def percent(self) -> float:
        return (self.current / self.total * 100) if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "current": self.current,
            "total": self.total,
            "percent": round(self.percent, 1),
            "stage": self.stage,
            "message": self.message,
        }


@dataclass
class IngestJob:
    """A background ingestion job."""
    id: str
    title: str
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    progress: JobProgress | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    # Internal
    _content: bytes = field(default=b"", repr=False)
    _metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress.to_dict() if self.progress else None,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    """In-memory storage for ingestion jobs."""

    def __init__(self, max_jobs: int = 100) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._max_jobs = max_jobs

    def create(
        self,
        title: str,
        content: bytes,
        metadata: dict[str, str] | None = None,
    ) -> IngestJob:
        """Create a new ingestion job."""
        job_id = str(uuid.uuid4())[:8]

        job = IngestJob(
            id=job_id,
            title=title,
            _content=content,
            _metadata=metadata or {},
        )

        # Evict oldest if at capacity
        if len(self._jobs) >= self._max_jobs:
            oldest_id = min(
                self._jobs.keys(),
                key=lambda k: self._jobs[k].created_at
            )
            del self._jobs[oldest_id]

        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> IngestJob | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[IngestJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def update(self, job_id: str, **kwargs) -> IngestJob | None:
        """Update job fields."""
        job = self.get(job_id)
        if not job:
            return None

        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)

        return job

    def delete(self, job_id: str) -> bool:
        """Delete a job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False


class AsyncIngestManager:
    """Manager for async document ingestion."""

    def __init__(
        self,
        pipeline: IngestPipeline,
        max_concurrent: int = 3,
    ) -> None:
        self._pipeline = pipeline
        self._job_store = JobStore()
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._on_progress: Callable[[str, JobProgress], None] | None = None

    def set_progress_callback(
        self,
        callback: Callable[[str, JobProgress], None],
    ) -> None:
        """Set callback for progress updates."""
        self._on_progress = callback

    async def submit(
        self,
        title: str,
        content: bytes,
        metadata: dict[str, str] | None = None,
    ) -> IngestJob:
        """Submit a new ingestion job and return immediately."""
        job = self._job_store.create(title, content, metadata)

        # Start processing in background
        asyncio.create_task(self._process_job(job.id))

        return job

    def submit_sync(
        self,
        title: str,
        content: bytes,
        metadata: dict[str, str] | None = None,
    ) -> IngestJob:
        """Submit a job synchronously (non-blocking, just queues it)."""
        job = self._job_store.create(title, content, metadata)
        return job

    async def _process_job(self, job_id: str) -> None:
        """Process a single job."""
        async with self._semaphore:
            job = self._job_store.get(job_id)
            if not job:
                return

            # Update status
            job.status = JobStatus.RUNNING
            job.started_at = time.time()

            try:
                # Report extraction stage
                job.progress = JobProgress(
                    current=0,
                    total=100,
                    stage="extracting",
                    message=f"Extracting text from {job.title}",
                )
                self._report_progress(job)

                # Run ingestion (blocking, but we're in async context with semaphore)
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._run_ingest,
                    job,
                )

                # Complete
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.progress = JobProgress(
                    current=100,
                    total=100,
                    stage="completed",
                    message="Ingestion complete",
                )
                self._report_progress(job)

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = time.time()
                job.progress = JobProgress(
                    current=0,
                    total=100,
                    stage="failed",
                    message=str(e),
                )
                self._report_progress(job)

    def _run_ingest(self, job: IngestJob) -> None:
        """Run the actual ingestion (synchronous)."""
        # Update progress
        job.progress = JobProgress(
            current=30,
            total=100,
            stage="processing",
            message="Processing document",
        )
        self._report_progress(job)

        # Call pipeline
        result = self._pipeline.ingest_file(
            title=job.title,
            content=job._content,
            metadata=job._metadata,
        )

        job.result = {
            "document_id": result.document_id,
            "title": result.title,
            "chunk_count": result.chunk_count,
        }

        # Update progress
        job.progress = JobProgress(
            current=80,
            total=100,
            stage="indexing",
            message="Building search index",
        )
        self._report_progress(job)

    def _report_progress(self, job: IngestJob) -> None:
        """Report progress via callback if set."""
        if self._on_progress and job.progress:
            with contextlib.suppress(Exception):
                self._on_progress(job.id, job.progress)

    def get_job(self, job_id: str) -> IngestJob | None:
        """Get job status."""
        return self._job_store.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[IngestJob]:
        """List jobs."""
        return self._job_store.list(status, limit)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job (running jobs can't be cancelled)."""
        job = self._job_store.get(job_id)
        if not job:
            return False

        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            return True

        return False


# Global manager instance
_manager: AsyncIngestManager | None = None


def get_ingest_manager(pipeline: IngestPipeline | None = None) -> AsyncIngestManager | None:
    """Get or create the global ingest manager."""
    global _manager
    if _manager is None and pipeline is not None:
        _manager = AsyncIngestManager(pipeline)
    return _manager


__all__ = [
    "JobStatus",
    "JobProgress",
    "IngestJob",
    "JobStore",
    "AsyncIngestManager",
    "get_ingest_manager",
]
