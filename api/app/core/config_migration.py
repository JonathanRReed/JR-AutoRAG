"""Config Migration: Upgrade configs from v1 to v2.0.

Provides automated migration for:
- Adding new v2.0 fields with sensible defaults
- Preserving existing user customizations
- Index versioning for re-index requirements
- Backward compatibility checks
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("autorag.migration")


@dataclass
class MigrationResult:
    """Result of a config migration."""
    success: bool
    version_from: str
    version_to: str
    changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_reindex: bool = False
    backup_path: str | None = None


# v2.0 new fields with defaults
V2_NEW_FIELDS = {
    # Phase 1
    "abstain_when_unverified": {
        "default": False,
        "description": "Abstain when evidence insufficient",
        "requires_reindex": False,
    },

    # Phase 2
    "self_rag_critic": {
        "default": False,
        "description": "Enable Self-RAG LLM critic",
        "requires_reindex": False,
    },
    "retrieval_cascade": {
        "default": False,
        "description": "Staged retrieval with stop criteria",
        "requires_reindex": False,
    },
    "auto_hybrid_weights": {
        "default": False,
        "description": "Automatic weight optimization",
        "requires_reindex": False,
    },

    # Phase 3
    "contextual_enrichment": {
        "default": False,
        "description": "Add document context to chunks",
        "requires_reindex": True,  # Changes chunk content
    },
    "multi_granularity": {
        "default": False,
        "description": "Multi-level indexing",
        "requires_reindex": True,  # Changes index structure
    },
    "structured_data_parsing": {
        "default": True,
        "description": "Parse JSON/CSV/YAML",
        "requires_reindex": True,  # Changes ingestion
    },

    # Phase 4
    "circuit_breakers": {
        "default": True,
        "description": "Enable circuit breakers",
        "requires_reindex": False,
    },
    "cost_tracking": {
        "default": True,
        "description": "Track API costs",
        "requires_reindex": False,
    },
    # LangExtract enrichment defaults
    "langextract_enabled": {
        "default": False,
        "description": "Enable ingestion-time LangExtract enrichment",
        "requires_reindex": False,
    },
    "langextract_profile_default": {
        "default": "generic_entities_v1",
        "description": "Default LangExtract profile",
        "requires_reindex": False,
    },
    "langextract_model_source": {
        "default": "gatherer",
        "description": "Model role used for extraction",
        "requires_reindex": False,
    },
    "langextract_timeout_sec": {
        "default": 20,
        "description": "LangExtract request timeout seconds",
        "requires_reindex": False,
    },
    "langextract_max_chars": {
        "default": 12000,
        "description": "Maximum source characters for extraction",
        "requires_reindex": False,
    },
    "langextract_max_synthetic_facts": {
        "default": 200,
        "description": "Maximum synthetic facts appended to ingestion text",
        "requires_reindex": False,
    },
}

# Preset upgrades for v2.0
V2_PRESET_UPGRADES = {
    "turbo": {
        "abstain_when_unverified": False,
        "self_rag_critic": False,
    },
    "fast": {
        "abstain_when_unverified": False,
        "self_rag_critic": False,
    },
    "balanced": {
        "abstain_when_unverified": True,
        "auto_hybrid_weights": True,
    },
    "thorough": {
        "abstain_when_unverified": True,
        "self_rag_critic": True,
        "retrieval_cascade": True,
    },
    "ultra_accurate": {
        "abstain_when_unverified": True,
        "self_rag_critic": True,
        "retrieval_cascade": True,
        "contextual_enrichment": True,
    },
}


class ConfigMigrator:
    """Migrate configs from v1 to v2.0."""

    CURRENT_VERSION = "2.0.0"

    def __init__(self, backup_dir: str | None = None) -> None:
        """Initialize migrator.

        Args:
            backup_dir: Directory for config backups (default: same as config)
        """
        self.backup_dir = Path(backup_dir) if backup_dir else None

    def get_config_version(self, config: dict[str, Any]) -> str:
        """Get version from config (default "1.0.0" for old configs)."""
        return config.get("version", "1.0.0")

    def needs_migration(self, config: dict[str, Any]) -> bool:
        """Check if config needs migration."""
        version = self.get_config_version(config)
        return version < self.CURRENT_VERSION

    def _backup_config(
        self,
        config: dict[str, Any],
        original_path: Path | None = None,
    ) -> str | None:
        """Create backup of original config."""
        if not self.backup_dir and not original_path:
            return None

        backup_dir = self.backup_dir or original_path.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"config_backup_{timestamp}.json"

        try:
            with open(backup_path, "w") as f:
                json.dump(config, f, indent=2)
            return str(backup_path)
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            return None

    def migrate(
        self,
        config: dict[str, Any],
        original_path: Path | None = None,
    ) -> tuple[dict[str, Any], MigrationResult]:
        """Migrate a config dict to v2.0.

        Args:
            config: Original config dictionary
            original_path: Path to original file (for backup)

        Returns:
            Tuple of (migrated_config, migration_result)
        """
        version_from = self.get_config_version(config)
        changes: list[str] = []
        warnings: list[str] = []
        requires_reindex = False

        # Create backup
        backup_path = self._backup_config(config, original_path)

        # Create new config (copy)
        new_config = config.copy()

        # Get retrieval section
        retrieval = new_config.get("retrieval", {})
        if isinstance(retrieval, dict):
            retrieval = retrieval.copy()
        else:
            retrieval = {}

        # Add new v2.0 fields
        for field_name, field_info in V2_NEW_FIELDS.items():
            if field_name not in retrieval:
                retrieval[field_name] = field_info["default"]
                changes.append(f"Added '{field_name}' = {field_info['default']}")

                if field_info["requires_reindex"]:
                    requires_reindex = True

        # Apply preset upgrades if a preset is set
        preset = new_config.get("preset", retrieval.get("preset", ""))
        if preset and preset.lower() in V2_PRESET_UPGRADES:
            upgrades = V2_PRESET_UPGRADES[preset.lower()]
            for field_name, value in upgrades.items():
                if retrieval.get(field_name) != value:
                    retrieval[field_name] = value
                    changes.append(f"Preset '{preset}': set '{field_name}' = {value}")

        # Update retrieval section
        new_config["retrieval"] = retrieval

        # Set version
        new_config["version"] = self.CURRENT_VERSION
        changes.append(f"Updated version to {self.CURRENT_VERSION}")

        # Add warnings for breaking changes
        if requires_reindex:
            warnings.append(
                "Some new features require re-indexing for full effect. "
                "Run re-index to enable: contextual_enrichment, multi_granularity, structured_data_parsing"
            )

        result = MigrationResult(
            success=True,
            version_from=version_from,
            version_to=self.CURRENT_VERSION,
            changes=changes,
            warnings=warnings,
            requires_reindex=requires_reindex,
            backup_path=backup_path,
        )

        return new_config, result

    def migrate_file(self, config_path: str | Path) -> MigrationResult:
        """Migrate a config file in place.

        Args:
            config_path: Path to config file

        Returns:
            MigrationResult
        """
        path = Path(config_path)

        if not path.exists():
            return MigrationResult(
                success=False,
                version_from="unknown",
                version_to=self.CURRENT_VERSION,
                warnings=[f"Config file not found: {path}"],
            )

        try:
            with open(path) as f:
                config = json.load(f)
        except Exception as e:
            return MigrationResult(
                success=False,
                version_from="unknown",
                version_to=self.CURRENT_VERSION,
                warnings=[f"Failed to read config: {e}"],
            )

        if not self.needs_migration(config):
            return MigrationResult(
                success=True,
                version_from=self.get_config_version(config),
                version_to=self.CURRENT_VERSION,
                changes=["No migration needed - already at v2.0"],
            )

        new_config, result = self.migrate(config, path)

        # Write migrated config
        try:
            with open(path, "w") as f:
                json.dump(new_config, f, indent=2)
            result.changes.append(f"Saved migrated config to {path}")
        except Exception as e:
            result.success = False
            result.warnings.append(f"Failed to write migrated config: {e}")

        return result


# Singleton
_migrator: ConfigMigrator | None = None


def get_config_migrator(backup_dir: str | None = None) -> ConfigMigrator:
    """Get or create the config migrator."""
    global _migrator
    if _migrator is None:
        _migrator = ConfigMigrator(backup_dir)
    return _migrator


def migrate_config(config: dict[str, Any]) -> tuple[dict[str, Any], MigrationResult]:
    """Convenience function to migrate a config dict."""
    return get_config_migrator().migrate(config)


__all__ = [
    "MigrationResult",
    "V2_NEW_FIELDS",
    "V2_PRESET_UPGRADES",
    "ConfigMigrator",
    "get_config_migrator",
    "migrate_config",
]
