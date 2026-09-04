"""Enhanced audit logging for enterprise compliance.

Provides persistent audit trail for:
- Query operations
- Document ingestion/deletion
- Configuration changes
- Authentication events

Designed for internal company hosting with file-based persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AuditAction(str, Enum):
    """Types of auditable actions."""

    QUERY = "query"
    INGEST = "ingest"
    DELETE = "delete"
    CONFIG_CHANGE = "config_change"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    EXPORT = "export"
    EVAL_RUN = "eval_run"
    SYSTEM = "system"


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: datetime
    action: AuditAction
    details: dict[str, Any]
    user_id: str | None = None
    ip_address: str | None = None
    success: bool = True
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "details": self.details,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            action=AuditAction(data["action"]),
            details=data.get("details", {}),
            user_id=data.get("user_id"),
            ip_address=data.get("ip_address"),
            success=data.get("success", True),
            duration_ms=data.get("duration_ms"),
        )


class AuditLog:
    """Persistent audit logging for enterprise compliance.

    Features:
    - File-based persistence (JSONL format)
    - Automatic log rotation by date
    - Query filtering by time, action, user
    - Export capabilities

    Usage:
        audit = AuditLog()
        audit.log(AuditEntry(
            timestamp=datetime.now(UTC),
            action=AuditAction.QUERY,
            details={"query": "...", "documents": [...]},
            user_id="user123",
        ))
    """

    DEFAULT_LOG_DIR = "data/audit"
    MAX_ENTRIES_IN_MEMORY = 1000

    def __init__(self, log_dir: str | Path | None = None) -> None:
        """Initialize audit log.

        Args:
            log_dir: Directory for log files. Defaults to data/audit.
        """
        self._log_dir = Path(log_dir or self.DEFAULT_LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffer for recent entries
        self._buffer: list[AuditEntry] = []
        self._current_date: str | None = None
        self._current_file: Path | None = None

    def _get_log_file(self, date: datetime | None = None) -> Path:
        """Get log file path for a given date."""
        if date is None:
            date = datetime.now(UTC)
        date_str = date.strftime("%Y-%m-%d")
        return self._log_dir / f"audit_{date_str}.jsonl"

    def _ensure_file_for_today(self) -> None:
        """Ensure we have the correct file handle for today."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if self._current_date != today:
            self._current_date = today
            self._current_file = self._get_log_file()

    def log(self, entry: AuditEntry) -> None:
        """Record an audit entry.

        Writes to disk immediately for durability.
        """
        self._ensure_file_for_today()

        # Write to file
        with open(self._current_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        # Add to buffer
        self._buffer.append(entry)

        # Trim buffer if too large
        if len(self._buffer) > self.MAX_ENTRIES_IN_MEMORY:
            self._buffer = self._buffer[-self.MAX_ENTRIES_IN_MEMORY :]

    def log_query(
        self,
        query: str,
        documents: list[str] | None = None,
        user_id: str | None = None,
        ip_address: str | None = None,
        success: bool = True,
        duration_ms: float | None = None,
    ) -> None:
        """Convenience method to log a query operation."""
        self.log(
            AuditEntry(
                timestamp=datetime.now(UTC),
                action=AuditAction.QUERY,
                details={
                    "query": query[:500],  # Truncate for storage
                    "documents": documents or [],
                },
                user_id=user_id,
                ip_address=ip_address,
                success=success,
                duration_ms=duration_ms,
            )
        )

    def log_ingest(
        self,
        document_id: str,
        filename: str,
        size_bytes: int,
        user_id: str | None = None,
        ip_address: str | None = None,
        success: bool = True,
    ) -> None:
        """Convenience method to log document ingestion."""
        self.log(
            AuditEntry(
                timestamp=datetime.now(UTC),
                action=AuditAction.INGEST,
                details={
                    "document_id": document_id,
                    "filename": filename,
                    "size_bytes": size_bytes,
                },
                user_id=user_id,
                ip_address=ip_address,
                success=success,
            )
        )

    def log_delete(
        self,
        document_id: str,
        user_id: str | None = None,
        ip_address: str | None = None,
        success: bool = True,
    ) -> None:
        """Convenience method to log document deletion."""
        self.log(
            AuditEntry(
                timestamp=datetime.now(UTC),
                action=AuditAction.DELETE,
                details={"document_id": document_id},
                user_id=user_id,
                ip_address=ip_address,
                success=success,
            )
        )

    def log_auth(
        self,
        success: bool,
        user_id: str | None = None,
        ip_address: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Log an authentication attempt."""
        action = AuditAction.AUTH_SUCCESS if success else AuditAction.AUTH_FAILURE
        self.log(
            AuditEntry(
                timestamp=datetime.now(UTC),
                action=action,
                details={"reason": reason} if reason else {},
                user_id=user_id,
                ip_address=ip_address,
                success=success,
            )
        )

    def query(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        action: AuditAction | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit log entries.

        Args:
            since: Start time filter
            until: End time filter
            action: Filter by action type
            user_id: Filter by user
            limit: Maximum entries to return

        Returns:
            List of matching entries (newest first)
        """
        # First check in-memory buffer
        entries = []
        for entry in reversed(self._buffer):
            if self._matches_filter(entry, since, until, action, user_id):
                entries.append(entry)
                if len(entries) >= limit:
                    return entries

        # If need more, read from files
        if len(entries) < limit:
            # Get list of log files
            log_files = sorted(self._log_dir.glob("audit_*.jsonl"), reverse=True)

            for log_file in log_files:
                # Check if file date is in range
                file_date_str = log_file.stem.replace("audit_", "")
                try:
                    file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                    if since and file_date.date() < since.date():
                        continue
                    if until and file_date.date() > until.date():
                        continue
                except ValueError:
                    continue

                # Read file entries
                file_entries = self._read_log_file(log_file)
                for entry in reversed(file_entries):
                    if entry not in self._buffer and self._matches_filter(
                        entry, since, until, action, user_id
                    ):  # Avoid duplicates
                        entries.append(entry)
                        if len(entries) >= limit:
                            return entries

        return entries

    def _matches_filter(
        self,
        entry: AuditEntry,
        since: datetime | None,
        until: datetime | None,
        action: AuditAction | None,
        user_id: str | None,
    ) -> bool:
        """Check if entry matches filter criteria."""
        if since and entry.timestamp < since:
            return False
        if until and entry.timestamp > until:
            return False
        if action and entry.action != action:
            return False
        return not (user_id and entry.user_id != user_id)

    def _read_log_file(self, path: Path) -> list[AuditEntry]:
        """Read entries from a log file."""
        entries = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            entries.append(AuditEntry.from_dict(data))
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
        except FileNotFoundError:
            pass
        return entries

    def export(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
        format: str = "json",
    ) -> str:
        """Export audit log entries.

        Args:
            since: Start time
            until: End time
            format: "json" or "csv"

        Returns:
            Formatted string of entries
        """
        entries = self.query(since=since, until=until, limit=10000)

        if format == "json":
            return json.dumps([e.to_dict() for e in entries], indent=2)
        elif format == "csv":
            lines = ["timestamp,action,user_id,ip_address,success,details"]
            for e in entries:
                details_str = json.dumps(e.details).replace('"', '""')
                lines.append(
                    f"{e.timestamp.isoformat()},{e.action.value},"
                    f"{e.user_id or ''},{e.ip_address or ''},"
                    f'{e.success},"{details_str}"'
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")

    def get_stats(
        self,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Get audit log statistics."""
        entries = self.query(since=since, limit=10000)

        by_action: dict[str, int] = {}
        by_user: dict[str, int] = {}
        success_count = 0
        failure_count = 0

        for entry in entries:
            by_action[entry.action.value] = by_action.get(entry.action.value, 0) + 1
            if entry.user_id:
                by_user[entry.user_id] = by_user.get(entry.user_id, 0) + 1
            if entry.success:
                success_count += 1
            else:
                failure_count += 1

        return {
            "total_entries": len(entries),
            "by_action": by_action,
            "by_user": by_user,
            "success_count": success_count,
            "failure_count": failure_count,
        }


# Singleton instance
_audit_instance: AuditLog | None = None


def get_audit_log() -> AuditLog:
    """Get the global audit log instance."""
    global _audit_instance
    if _audit_instance is None:
        _audit_instance = AuditLog()
    return _audit_instance


__all__ = [
    "AuditAction",
    "AuditEntry",
    "AuditLog",
    "get_audit_log",
]
