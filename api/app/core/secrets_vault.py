"""Secure secrets storage for API keys and credentials.

Provides secure storage options for sensitive configuration:
- OS keychain integration (macOS Keychain, Windows Credential Store, Linux Secret Service)
- Encrypted local vault file as fallback
- Automatic migration from plain config

Never stores secrets in plain JSON config files.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Try to import keyring for OS-level secret storage
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

# Try to import cryptography for encrypted vault
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

SERVICE_NAME = "jr-autorag"
VAULT_FILENAME = "secrets.vault"
VAULT_KEY_ENV = "AUTORAG_VAULT_KEY"


@dataclass
class SecretMetadata:
    """Metadata about a stored secret."""
    key: str
    created_at: datetime
    updated_at: datetime
    source: str  # "keychain", "vault", "env"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
        }


# =============================================================================
# Keychain Backend
# =============================================================================

class KeychainBackend:
    """OS keychain backend for secret storage."""
    
    def __init__(self, service_name: str = SERVICE_NAME) -> None:
        if not KEYRING_AVAILABLE:
            raise RuntimeError("keyring package not installed. Install with: pip install keyring")
        self.service_name = service_name
    
    def get(self, key: str) -> Optional[str]:
        """Get a secret from the keychain."""
        try:
            return keyring.get_password(self.service_name, key)
        except Exception:
            return None
    
    def set(self, key: str, value: str) -> None:
        """Store a secret in the keychain."""
        keyring.set_password(self.service_name, key, value)
    
    def delete(self, key: str) -> bool:
        """Delete a secret from the keychain."""
        try:
            keyring.delete_password(self.service_name, key)
            return True
        except Exception:
            return False
    
    def available(self) -> bool:
        """Check if keychain is available."""
        try:
            # Try a simple operation to verify keychain works
            test_key = "__autorag_keychain_test__"
            keyring.set_password(self.service_name, test_key, "test")
            keyring.delete_password(self.service_name, test_key)
            return True
        except Exception:
            return False


# =============================================================================
# Encrypted Vault Backend
# =============================================================================

class EncryptedVaultBackend:
    """Encrypted local file vault for secret storage."""
    
    def __init__(
        self,
        vault_path: Optional[Path] = None,
        encryption_key: Optional[str] = None,
    ) -> None:
        if not CRYPTO_AVAILABLE:
            raise RuntimeError(
                "cryptography package not installed. Install with: pip install cryptography"
            )
        
        self.vault_path = vault_path or Path("data/secrets.vault")
        self._fernet: Optional[Fernet] = None
        self._data: dict[str, str] = {}
        self._metadata: dict[str, SecretMetadata] = {}
        
        # Initialize encryption
        self._init_encryption(encryption_key)
        self._load()
    
    def _init_encryption(self, key: Optional[str] = None) -> None:
        """Initialize Fernet encryption with key."""
        if key:
            # Use provided key
            key_bytes = key.encode()
        elif os.environ.get(VAULT_KEY_ENV):
            # Use key from environment
            key_bytes = os.environ[VAULT_KEY_ENV].encode()
        else:
            # Generate and store a key (first run)
            key_path = self.vault_path.parent / ".vault_key"
            if key_path.exists():
                key_bytes = key_path.read_bytes()
            else:
                key_bytes = Fernet.generate_key()
                key_path.parent.mkdir(parents=True, exist_ok=True)
                # Set restrictive permissions
                key_path.write_bytes(key_bytes)
                try:
                    os.chmod(key_path, 0o600)
                except OSError:
                    pass  # May fail on Windows
        
        # Derive a proper Fernet key if needed
        if len(key_bytes) != 44:  # Fernet keys are 44 bytes base64
            # Use PBKDF2 to derive a proper key
            salt = b"jr-autorag-vault-v1"  # Static salt is OK here
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,
            )
            derived = base64.urlsafe_b64encode(kdf.derive(key_bytes))
            self._fernet = Fernet(derived)
        else:
            self._fernet = Fernet(key_bytes)
    
    def _load(self) -> None:
        """Load vault from disk."""
        if not self.vault_path.exists():
            self._data = {}
            self._metadata = {}
            return
        
        try:
            encrypted = self.vault_path.read_bytes()
            decrypted = self._fernet.decrypt(encrypted)
            vault_data = json.loads(decrypted.decode())
            self._data = vault_data.get("secrets", {})
            self._metadata = {
                k: SecretMetadata(
                    key=k,
                    created_at=datetime.fromisoformat(v["created_at"]),
                    updated_at=datetime.fromisoformat(v["updated_at"]),
                    source="vault",
                )
                for k, v in vault_data.get("metadata", {}).items()
            }
        except Exception as e:
            print(f"Warning: Could not load vault: {e}")
            self._data = {}
            self._metadata = {}
    
    def _save(self) -> None:
        """Save vault to disk."""
        vault_data = {
            "secrets": self._data,
            "metadata": {
                k: {
                    "created_at": v.created_at.isoformat(),
                    "updated_at": v.updated_at.isoformat(),
                }
                for k, v in self._metadata.items()
            },
        }
        
        serialized = json.dumps(vault_data).encode()
        encrypted = self._fernet.encrypt(serialized)
        
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        self.vault_path.write_bytes(encrypted)
        
        # Set restrictive permissions
        try:
            os.chmod(self.vault_path, 0o600)
        except OSError:
            pass
    
    def get(self, key: str) -> Optional[str]:
        """Get a secret from the vault."""
        return self._data.get(key)
    
    def set(self, key: str, value: str) -> None:
        """Store a secret in the vault."""
        now = datetime.utcnow()
        
        if key in self._metadata:
            self._metadata[key].updated_at = now
        else:
            self._metadata[key] = SecretMetadata(
                key=key,
                created_at=now,
                updated_at=now,
                source="vault",
            )
        
        self._data[key] = value
        self._save()
    
    def delete(self, key: str) -> bool:
        """Delete a secret from the vault."""
        if key in self._data:
            del self._data[key]
            del self._metadata[key]
            self._save()
            return True
        return False
    
    def list_keys(self) -> list[str]:
        """List all secret keys."""
        return list(self._data.keys())
    
    def get_metadata(self, key: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret."""
        return self._metadata.get(key)


# =============================================================================
# Unified Secrets Vault
# =============================================================================

class SecretsVault:
    """Unified secrets vault with fallback between backends.
    
    Priority:
    1. Environment variables (highest priority, read-only)
    2. OS keychain (if available and working)
    3. Encrypted local vault (fallback)
    
    Usage:
        vault = SecretsVault()
        vault.set("OPENAI_API_KEY", "sk-...")
        key = vault.get("OPENAI_API_KEY")
    """
    
    # Standard secret keys used by the application
    KNOWN_SECRETS = {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "COHERE_API_KEY",
        "HUGGINGFACE_TOKEN",
        "OLLAMA_API_KEY",
        "LM_STUDIO_API_KEY",
    }
    
    def __init__(
        self,
        prefer_keychain: bool = True,
        vault_path: Optional[Path] = None,
    ) -> None:
        """Initialize the secrets vault.
        
        Args:
            prefer_keychain: If True, try OS keychain first
            vault_path: Custom path for encrypted vault file
        """
        self._keychain: Optional[KeychainBackend] = None
        self._vault: Optional[EncryptedVaultBackend] = None
        
        # Try to initialize keychain backend
        if prefer_keychain and KEYRING_AVAILABLE:
            try:
                backend = KeychainBackend()
                if backend.available():
                    self._keychain = backend
            except Exception:
                pass
        
        # Initialize encrypted vault as fallback
        if CRYPTO_AVAILABLE:
            try:
                self._vault = EncryptedVaultBackend(vault_path=vault_path)
            except Exception as e:
                print(f"Warning: Could not initialize encrypted vault: {e}")
        
        if not self._keychain and not self._vault:
            print("Warning: No secure secret storage available. Secrets will only be read from environment variables.")
    
    def get(self, key: str) -> Optional[str]:
        """Get a secret value.
        
        Priority: env var > keychain > vault
        """
        # Check environment first (highest priority)
        env_value = os.environ.get(key)
        if env_value:
            return env_value
        
        # Try keychain
        if self._keychain:
            value = self._keychain.get(key)
            if value:
                return value
        
        # Try vault
        if self._vault:
            value = self._vault.get(key)
            if value:
                return value
        
        return None
    
    def set(self, key: str, value: str) -> None:
        """Store a secret value.
        
        Stores in keychain if available, otherwise vault.
        """
        if self._keychain:
            self._keychain.set(key, value)
        elif self._vault:
            self._vault.set(key, value)
        else:
            raise RuntimeError("No secure storage backend available")
    
    def delete(self, key: str) -> bool:
        """Delete a secret value."""
        deleted = False
        
        if self._keychain:
            deleted = self._keychain.delete(key) or deleted
        
        if self._vault:
            deleted = self._vault.delete(key) or deleted
        
        return deleted
    
    def list_keys(self) -> list[str]:
        """List all stored secret keys."""
        keys = set()
        
        # From vault
        if self._vault:
            keys.update(self._vault.list_keys())
        
        # From environment (known keys only)
        for key in self.KNOWN_SECRETS:
            if os.environ.get(key):
                keys.add(key)
        
        return sorted(keys)
    
    def get_source(self, key: str) -> Optional[str]:
        """Get the source of a secret (env/keychain/vault)."""
        if os.environ.get(key):
            return "env"
        
        if self._keychain and self._keychain.get(key):
            return "keychain"
        
        if self._vault and self._vault.get(key):
            return "vault"
        
        return None
    
    def migrate_from_config(self, config: dict[str, Any]) -> int:
        """Migrate secrets from plain config to vault.
        
        Returns number of secrets migrated.
        """
        migrated = 0
        
        for key in self.KNOWN_SECRETS:
            # Check various config locations
            value = config.get(key) or config.get(key.lower())
            if value and not self.get(key):
                self.set(key, value)
                migrated += 1
        
        # Check nested provider config
        providers = config.get("providers", {})
        for provider, settings in providers.items():
            if isinstance(settings, dict):
                api_key = settings.get("api_key") or settings.get("apiKey")
                if api_key:
                    key_name = f"{provider.upper()}_API_KEY"
                    if not self.get(key_name):
                        self.set(key_name, api_key)
                        migrated += 1
        
        return migrated
    
    def status(self) -> dict[str, Any]:
        """Get vault status information."""
        return {
            "keychain_available": self._keychain is not None,
            "vault_available": self._vault is not None,
            "stored_keys": len(self.list_keys()),
            "backend": "keychain" if self._keychain else ("vault" if self._vault else "env_only"),
        }


# =============================================================================
# Singleton
# =============================================================================

_vault_instance: Optional[SecretsVault] = None


def get_secrets_vault() -> SecretsVault:
    """Get the global secrets vault instance."""
    global _vault_instance
    if _vault_instance is None:
        _vault_instance = SecretsVault()
    return _vault_instance


__all__ = [
    "SecretsVault",
    "SecretMetadata",
    "KeychainBackend",
    "EncryptedVaultBackend",
    "get_secrets_vault",
]
