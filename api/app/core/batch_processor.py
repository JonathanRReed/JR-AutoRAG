"""Batch processing for parallel document ingestion.

This module provides:
- Parallel document processing
- Progress tracking
- Error handling and retry
- Throughput optimization
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypeVar
import threading


T = TypeVar("T")


class BatchStatus(str, Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    """A single item in a batch."""
    id: str
    data: Any
    status: BatchStatus = BatchStatus.PENDING
    result: Any = None
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    retries: int = 0
    
    @property
    def duration_ms(self) -> float:
        if not self.start_time or not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000


@dataclass
class BatchProgress:
    """Progress information for a batch job."""
    total: int
    completed: int
    failed: int
    pending: int
    current_item: str | None
    elapsed_ms: float
    estimated_remaining_ms: float
    
    @property
    def percent_complete(self) -> float:
        return (self.completed + self.failed) / self.total * 100 if self.total > 0 else 0


@dataclass
class BatchResult:
    """Result of a batch job."""
    job_id: str
    status: BatchStatus
    total: int
    successful: int
    failed: int
    items: list[BatchItem]
    start_time: datetime
    end_time: datetime | None
    errors: list[str] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time).total_seconds() * 1000


class BatchProcessor:
    """Processes items in parallel batches."""
    
    def __init__(
        self,
        max_workers: int = 4,
        batch_size: int = 10,
        max_retries: int = 2,
    ) -> None:
        self._max_workers = max_workers
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._jobs: dict[str, BatchResult] = {}
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()
    
    def _process_item(
        self,
        item: BatchItem,
        processor: Callable[[Any], Any],
        job_id: str,
    ) -> BatchItem:
        """Process a single item."""
        if job_id in self._cancelled:
            item.status = BatchStatus.CANCELLED
            return item
        
        item.status = BatchStatus.RUNNING
        item.start_time = datetime.now(timezone.utc)
        
        try:
            item.result = processor(item.data)
            item.status = BatchStatus.COMPLETED
        except Exception as e:
            item.error = str(e)
            item.retries += 1
            
            if item.retries <= self._max_retries:
                # Retry
                try:
                    item.result = processor(item.data)
                    item.status = BatchStatus.COMPLETED
                    item.error = None
                except Exception as retry_e:
                    item.error = str(retry_e)
                    item.status = BatchStatus.FAILED
            else:
                item.status = BatchStatus.FAILED
        
        item.end_time = datetime.now(timezone.utc)
        return item
    
    def process(
        self,
        items: list[Any],
        processor: Callable[[Any], Any],
        job_id: str | None = None,
        progress_callback: Callable[[BatchProgress], None] | None = None,
    ) -> BatchResult:
        """Process items in parallel.
        
        Args:
            items: Items to process
            processor: Function to apply to each item
            job_id: Optional job ID (auto-generated if not provided)
            progress_callback: Optional callback for progress updates
        
        Returns:
            BatchResult with all processed items
        """
        import uuid
        
        job_id = job_id or str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        # Create batch items
        batch_items = [
            BatchItem(id=f"{job_id}_{i}", data=item)
            for i, item in enumerate(items)
        ]
        
        # Initialize result
        result = BatchResult(
            job_id=job_id,
            status=BatchStatus.RUNNING,
            total=len(batch_items),
            successful=0,
            failed=0,
            items=batch_items,
            start_time=start_time,
            end_time=None,
        )
        
        with self._lock:
            self._jobs[job_id] = result
        
        completed = 0
        failed = 0
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._process_item, item, processor, job_id): item
                for item in batch_items
            }
            
            for future in as_completed(futures):
                item = futures[future]
                
                try:
                    processed_item = future.result()
                    
                    if processed_item.status == BatchStatus.COMPLETED:
                        completed += 1
                    elif processed_item.status == BatchStatus.FAILED:
                        failed += 1
                        result.errors.append(f"{item.id}: {processed_item.error}")
                except Exception as e:
                    failed += 1
                    item.status = BatchStatus.FAILED
                    item.error = str(e)
                    result.errors.append(f"{item.id}: {str(e)}")
                
                # Update progress
                if progress_callback:
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    rate = (completed + failed) / elapsed if elapsed > 0 else 0
                    remaining = len(batch_items) - completed - failed
                    est_remaining = remaining / rate if rate > 0 else 0
                    
                    progress_callback(BatchProgress(
                        total=len(batch_items),
                        completed=completed,
                        failed=failed,
                        pending=remaining,
                        current_item=item.id,
                        elapsed_ms=elapsed,
                        estimated_remaining_ms=est_remaining,
                    ))
        
        # Finalize result
        result.successful = completed
        result.failed = failed
        result.end_time = datetime.now(timezone.utc)
        result.status = (
            BatchStatus.COMPLETED if failed == 0
            else BatchStatus.FAILED if completed == 0
            else BatchStatus.COMPLETED  # Partial success
        )
        
        return result
    
    async def process_async(
        self,
        items: list[Any],
        processor: Callable[[Any], Any],
        job_id: str | None = None,
    ) -> BatchResult:
        """Async version of process."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.process(items, processor, job_id),
        )
    
    def cancel(self, job_id: str) -> bool:
        """Cancel a running job."""
        self._cancelled.add(job_id)
        
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = BatchStatus.CANCELLED
                return True
        return False
    
    def get_job(self, job_id: str) -> BatchResult | None:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def get_progress(self, job_id: str) -> BatchProgress | None:
        """Get current progress for a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            
            completed = sum(1 for i in job.items if i.status == BatchStatus.COMPLETED)
            failed = sum(1 for i in job.items if i.status == BatchStatus.FAILED)
            pending = sum(1 for i in job.items if i.status == BatchStatus.PENDING)
            running = [i.id for i in job.items if i.status == BatchStatus.RUNNING]
            
            elapsed = (datetime.now(timezone.utc) - job.start_time).total_seconds() * 1000
            rate = (completed + failed) / elapsed if elapsed > 0 else 0
            remaining = pending + len(running)
            est_remaining = remaining / rate * 1000 if rate > 0 else 0
            
            return BatchProgress(
                total=job.total,
                completed=completed,
                failed=failed,
                pending=pending,
                current_item=running[0] if running else None,
                elapsed_ms=elapsed,
                estimated_remaining_ms=est_remaining,
            )


__all__ = [
    "BatchStatus",
    "BatchItem",
    "BatchProgress",
    "BatchResult",
    "BatchProcessor",
]
