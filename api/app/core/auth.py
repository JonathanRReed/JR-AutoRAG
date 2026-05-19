"""Simple API key authentication for internal/local hosting.

Provides basic authentication without complex OAuth/SSO:
- API key validation
- Optional enforcement (disabled by default for local use)
- Environment-based configuration

Designed for local or internal company hosting, not public deployments.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class APIKey:
    """An API key with metadata."""
    key_hash: str  # SHA-256 hash of the key
    name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used: datetime | None = None
    enabled: bool = True
    scopes: list[str] = field(default_factory=list)  # e.g., ["read", "write", "admin"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_hash": self.key_hash,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "enabled": self.enabled,
            "scopes": self.scopes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APIKey:
        return cls(
            key_hash=data["key_hash"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            enabled=data.get("enabled", True),
            scopes=data.get("scopes", []),
        )


class APIKeyAuth:
    """Simple API key authentication for internal use.

    Features:
    - Keys are stored as SHA-256 hashes (not plaintext)
    - Optional enforcement (can be disabled for local dev)
    - Scope-based access control
    - Environment variable configuration

    Usage:
        auth = APIKeyAuth()
        if auth.require_auth():
            if not auth.verify(provided_key, required_scope="read"):
                raise AuthError("Invalid API key")
    """

    # Environment variable for enabling auth
    ENV_AUTH_ENABLED = "AUTORAG_AUTH_ENABLED"
    # Environment variable for admin key (comma-separated for multiple)
    ENV_API_KEYS = "AUTORAG_API_KEYS"

    def __init__(self, enabled: bool | None = None) -> None:
        """Initialize authentication.

        Args:
            enabled: Override for auth enabled state. If None, uses env var.
        """
        if enabled is not None:
            self._enabled = enabled
        else:
            env_val = os.environ.get(self.ENV_AUTH_ENABLED, "false").lower()
            self._enabled = env_val in ("true", "1", "yes")

        self._keys: dict[str, APIKey] = {}
        self._load_from_env()

    def _hash_key(self, key: str) -> str:
        """Create SHA-256 hash of API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _load_from_env(self) -> None:
        """Load API keys from environment variable."""
        env_keys = os.environ.get(self.ENV_API_KEYS, "")
        if not env_keys:
            return

        for i, key in enumerate(env_keys.split(",")):
            key = key.strip()
            if key:
                key_hash = self._hash_key(key)
                self._keys[key_hash] = APIKey(
                    key_hash=key_hash,
                    name=f"env_key_{i}",
                    scopes=["read", "write", "admin"],
                )

    @property
    def enabled(self) -> bool:
        """Check if authentication is enabled."""
        return self._enabled

    def has_keys(self) -> bool:
        """Return True if any API keys are configured."""
        return len(self._keys) > 0

    def require_auth(self) -> bool:
        """Check if authentication is required.

        Returns True if auth is enabled.
        """
        return self._enabled

    def _scope_allows(self, scopes: list[str], required_scope: str | None) -> bool:
        if required_scope is None:
            return True
        if "admin" in scopes:
            return True
        if required_scope == "read":
            return "read" in scopes or "write" in scopes
        if required_scope == "write":
            return "write" in scopes
        return required_scope in scopes

    def generate_key(self, name: str, scopes: list[str] | None = None) -> tuple[str, APIKey]:
        """Generate a new API key.

        Returns:
            Tuple of (plaintext_key, APIKey object)

        Note: The plaintext key is only returned once and should be
        given to the user. Only the hash is stored.
        """
        plaintext = secrets.token_urlsafe(32)
        key_hash = self._hash_key(plaintext)

        api_key = APIKey(
            key_hash=key_hash,
            name=name,
            scopes=scopes or ["read"],
        )
        self._keys[key_hash] = api_key

        return plaintext, api_key

    def verify(
        self,
        key: str,
        required_scope: str | None = None,
    ) -> bool:
        """Verify an API key.

        Args:
            key: Plaintext API key to verify
            required_scope: Optional scope that must be present

        Returns:
            True if key is valid and has required scope
        """
        if not self._enabled:
            return True  # Auth disabled, allow all

        key_hash = self._hash_key(key)
        api_key = self._keys.get(key_hash)

        if not api_key:
            return False

        if not api_key.enabled:
            return False

        if not self._scope_allows(api_key.scopes, required_scope):
            return False

        # Update last used
        api_key.last_used = datetime.now(UTC)

        return True

    def get_scopes(self, key: str) -> list[str]:
        """Return scopes for a plaintext key, or empty list if unknown."""
        key_hash = self._hash_key(key)
        api_key = self._keys.get(key_hash)
        if not api_key:
            return []
        return list(api_key.scopes)

    def revoke(self, key: str) -> bool:
        """Revoke an API key (disable it)."""
        key_hash = self._hash_key(key)
        if key_hash in self._keys:
            self._keys[key_hash].enabled = False
            return True
        return False

    def delete(self, key: str) -> bool:
        """Delete an API key entirely."""
        key_hash = self._hash_key(key)
        if key_hash in self._keys:
            del self._keys[key_hash]
            return True
        return False

    def list_keys(self) -> list[dict[str, Any]]:
        """List all API keys (without hashes for security)."""
        return [
            {
                "name": k.name,
                "created_at": k.created_at.isoformat(),
                "last_used": k.last_used.isoformat() if k.last_used else None,
                "enabled": k.enabled,
                "scopes": k.scopes,
            }
            for k in self._keys.values()
        ]


# Singleton instance
_auth_instance: APIKeyAuth | None = None


def get_auth() -> APIKeyAuth:
    """Get the global authentication instance."""
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = APIKeyAuth()
    return _auth_instance


__all__ = [
    "APIKey",
    "APIKeyAuth",
    "get_auth",
]
