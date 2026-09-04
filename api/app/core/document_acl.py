"""Document-level access control.

Implements access control lists (ACLs) for documents:
- Per-document owner, readers, and writers
- Query-time filtering based on user permissions
- Integration with authentication system
- Audit logging for access attempts

This ensures users only see documents they're authorized to access.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk

logger = logging.getLogger("autorag.acl")


def _env_flag(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"true", "1", "yes", "on"}


def resolve_acl_defaults(auth_enabled: bool) -> tuple[bool, bool]:
    """Return (default_public, new_doc_public) based on env and auth state."""
    default_public = _env_flag("AUTORAG_ACL_DEFAULT_PUBLIC", True)
    new_doc_public = _env_flag("AUTORAG_ACL_NEW_DOC_PUBLIC", not auth_enabled)
    return default_public, new_doc_public


# =============================================================================
# ACL Types
# =============================================================================


@dataclass
class DocumentACL:
    """Access control list for a document."""

    document_id: str
    owner: str  # User ID who owns the document
    readers: list[str] = field(default_factory=list)  # User IDs or "*" for public
    writers: list[str] = field(default_factory=list)  # User IDs who can modify
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_public(self) -> bool:
        """Check if document is publicly readable."""
        return "*" in self.readers

    def can_read(self, user_id: str) -> bool:
        """Check if user can read this document."""
        if self.is_public:
            return True
        if user_id == self.owner:
            return True
        return user_id in self.readers or user_id in self.writers

    def can_write(self, user_id: str) -> bool:
        """Check if user can modify this document."""
        if user_id == self.owner:
            return True
        return user_id in self.writers

    def add_reader(self, user_id: str) -> None:
        """Add a reader."""
        if user_id not in self.readers:
            self.readers.append(user_id)
            self.updated_at = datetime.now(UTC)

    def add_writer(self, user_id: str) -> None:
        """Add a writer (writers can also read)."""
        if user_id not in self.writers:
            self.writers.append(user_id)
            self.updated_at = datetime.now(UTC)

    def remove_reader(self, user_id: str) -> bool:
        """Remove a reader."""
        if user_id in self.readers:
            self.readers.remove(user_id)
            self.updated_at = datetime.now(UTC)
            return True
        return False

    def remove_writer(self, user_id: str) -> bool:
        """Remove a writer."""
        if user_id in self.writers:
            self.writers.remove(user_id)
            self.updated_at = datetime.now(UTC)
            return True
        return False

    def make_public(self) -> None:
        """Make document publicly readable."""
        if "*" not in self.readers:
            self.readers.append("*")
            self.updated_at = datetime.now(UTC)

    def make_private(self) -> None:
        """Make document private (owner-only)."""
        self.readers = [r for r in self.readers if r != "*"]
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "owner": self.owner,
            "readers": self.readers,
            "writers": self.writers,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_public": self.is_public,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentACL:
        return cls(
            document_id=data["document_id"],
            owner=data["owner"],
            readers=data.get("readers", []),
            writers=data.get("writers", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(UTC),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(UTC),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def create_public(cls, document_id: str, owner: str) -> DocumentACL:
        """Create a public document ACL."""
        return cls(
            document_id=document_id,
            owner=owner,
            readers=["*"],
        )

    @classmethod
    def create_private(cls, document_id: str, owner: str) -> DocumentACL:
        """Create a private document ACL."""
        return cls(
            document_id=document_id,
            owner=owner,
        )


# =============================================================================
# ACL Store
# =============================================================================


class ACLStore:
    """Persistent storage for document ACLs."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path("data/document_acls.json")
        self._acls: dict[str, DocumentACL] = {}
        self._load()

    def _load(self) -> None:
        """Load ACLs from disk."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self._acls = {
                    doc_id: DocumentACL.from_dict(acl) for doc_id, acl in data.items()
                }
            except Exception as e:
                logger.warning(f"Failed to load ACLs: {e}")
                self._acls = {}

    def _save(self) -> None:
        """Save ACLs to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(
                {doc_id: acl.to_dict() for doc_id, acl in self._acls.items()},
                f,
                indent=2,
            )

    def get(self, document_id: str) -> DocumentACL | None:
        """Get ACL for a document."""
        return self._acls.get(document_id)

    def set(self, acl: DocumentACL) -> None:
        """Set ACL for a document."""
        self._acls[acl.document_id] = acl
        self._save()

    def delete(self, document_id: str) -> bool:
        """Delete ACL for a document."""
        if document_id in self._acls:
            del self._acls[document_id]
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Delete all ACLs."""
        self._acls = {}
        self._save()

    def list_by_owner(self, owner: str) -> list[DocumentACL]:
        """List all ACLs owned by a user."""
        return [acl for acl in self._acls.values() if acl.owner == owner]

    def list_readable_by(self, user_id: str) -> list[str]:
        """List all document IDs readable by a user."""
        return [doc_id for doc_id, acl in self._acls.items() if acl.can_read(user_id)]


# =============================================================================
# ACL Enforcer
# =============================================================================


class ACLEnforcer:
    """Enforce document access controls at query time.

    Usage:
        enforcer = ACLEnforcer()
        accessible = enforcer.filter_by_access(chunks, user_id="user123")
    """

    def __init__(
        self,
        store: ACLStore | None = None,
        default_public: bool = True,  # If no ACL exists, treat as public
    ) -> None:
        self.store = store or ACLStore()
        self.default_public = default_public

    def check_access(
        self,
        document_id: str,
        user_id: str | None,
        action: str = "read",
    ) -> tuple[bool, str]:
        """Check if user can access a document.

        Args:
            document_id: Document to check
            user_id: User requesting access
            action: "read" or "write"

        Returns:
            Tuple of (allowed, reason)
        """
        acl = self.store.get(document_id)

        if acl is None:
            if action == "write":
                if not user_id and self.default_public:
                    return (
                        True,
                        "No ACL defined, default public write (unauthenticated)",
                    )
                return False, "No ACL defined for write access"
            if self.default_public:
                return True, "No ACL defined, default public access"
            return False, "No ACL defined, default private"

        if action == "write":
            if not user_id:
                return False, "Missing user context for write access"
            if acl.can_write(user_id):
                return True, "User has write access"
            return False, "User lacks write permission"

        # Default to read check
        if not user_id:
            if acl.is_public:
                return True, "Public document"
            return False, "Missing user context for private document"
        if acl.can_read(user_id):
            return True, "User has read access"
        return False, "User lacks read permission"

    def filter_by_access(
        self,
        chunks: list[EvidenceChunk],
        user_id: str | None,
    ) -> list[EvidenceChunk]:
        """Filter chunks to only those the user can access.

        Args:
            chunks: Retrieved evidence chunks
            user_id: User requesting access

        Returns:
            Filtered list of accessible chunks
        """
        if not user_id:
            if self.default_public:
                return chunks
            # Allow only explicitly public docs when default is private.
            accessible = []
            for chunk in chunks:
                doc_id = self._get_document_id(chunk)
                if not doc_id:
                    continue
                acl = self.store.get(doc_id)
                if acl and acl.is_public:
                    accessible.append(chunk)
            return accessible

        accessible = []
        denied_count = 0

        for chunk in chunks:
            # Extract document ID from chunk
            doc_id = self._get_document_id(chunk)
            if doc_id is None:
                # No document ID - use default behavior
                if self.default_public:
                    accessible.append(chunk)
                continue

            allowed, _ = self.check_access(doc_id, user_id, "read")
            if allowed:
                accessible.append(chunk)
            else:
                denied_count += 1

        if denied_count > 0:
            logger.info(f"ACL filtered out {denied_count} chunks for user {user_id}")

        return accessible

    def _get_document_id(self, chunk: EvidenceChunk) -> str | None:
        """Extract document ID from a chunk."""
        # Try various attributes
        if hasattr(chunk, "document_id"):
            return chunk.document_id
        if hasattr(chunk, "doc_id"):
            return chunk.doc_id
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            return chunk.metadata.get("document_id") or chunk.metadata.get("doc_id")
        return None

    def create_acl_for_document(
        self,
        document_id: str,
        owner: str,
        public: bool = True,
    ) -> DocumentACL:
        """Create an ACL for a new document.

        Args:
            document_id: Document ID
            owner: User ID of the owner
            public: Whether to make public by default

        Returns:
            Created ACL
        """
        if public:
            acl = DocumentACL.create_public(document_id, owner)
        else:
            acl = DocumentACL.create_private(document_id, owner)

        self.store.set(acl)
        return acl


# =============================================================================
# Singleton
# =============================================================================

_acl_store: ACLStore | None = None
_acl_enforcer: ACLEnforcer | None = None


def get_acl_store() -> ACLStore:
    """Get the global ACL store."""
    global _acl_store
    if _acl_store is None:
        _acl_store = ACLStore()
    return _acl_store


def get_acl_enforcer(default_public: bool | None = None) -> ACLEnforcer:
    """Get the global ACL enforcer."""
    global _acl_enforcer
    if _acl_enforcer is None:
        initial_default = default_public if default_public is not None else True
        _acl_enforcer = ACLEnforcer(
            store=get_acl_store(), default_public=initial_default
        )
    elif default_public is not None:
        _acl_enforcer.default_public = default_public
    return _acl_enforcer


__all__ = [
    "DocumentACL",
    "ACLStore",
    "ACLEnforcer",
    "get_acl_store",
    "get_acl_enforcer",
    "resolve_acl_defaults",
]
