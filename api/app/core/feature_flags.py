"""Feature Flags: Controlled rollout of v2.0 features.

Provides a centralized feature flag system for:
- Gradual rollout of risky features
- A/B testing capabilities
- Kill switches for production issues
- Environment-specific configuration

Feature flags can be set via environment variables, config files, or API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class FeatureState(str, Enum):
    """State of a feature flag."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    PERCENTAGE = "percentage"  # Enabled for X% of requests


class RiskLevel(str, Enum):
    """Risk level for feature rollout."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeatureFlag:
    """A single feature flag definition."""
    name: str
    description: str
    default: bool = False
    risk_level: RiskLevel = RiskLevel.MEDIUM
    percentage: float = 100.0  # When state is PERCENTAGE
    env_var: str = ""  # Override via environment variable
    dependencies: list[str] = field(default_factory=list)  # Other flags this depends on
    metadata: dict[str, Any] = field(default_factory=dict)


# v2.0 Feature Flag Definitions
V2_FEATURE_FLAGS: dict[str, FeatureFlag] = {
    # Phase 1 - Foundation
    "FF_ABSTENTION": FeatureFlag(
        name="FF_ABSTENTION",
        description="Abstain from answering when evidence is insufficient",
        default=True,
        risk_level=RiskLevel.LOW,
        env_var="FF_ABSTENTION",
    ),
    "FF_TRUST_SCORE": FeatureFlag(
        name="FF_TRUST_SCORE",
        description="Show trust/confidence score in UI",
        default=True,
        risk_level=RiskLevel.LOW,
        env_var="FF_TRUST_SCORE",
    ),
    
    # Phase 2 - Agentic Loops
    "FF_SELF_RAG": FeatureFlag(
        name="FF_SELF_RAG",
        description="Self-RAG LLM-based critic for response quality",
        default=False,
        risk_level=RiskLevel.HIGH,
        env_var="FF_SELF_RAG",
        dependencies=["FF_ABSTENTION"],
    ),
    "FF_RETRIEVAL_CASCADE": FeatureFlag(
        name="FF_RETRIEVAL_CASCADE",
        description="Staged retrieval with early stopping",
        default=True,
        risk_level=RiskLevel.MEDIUM,
        env_var="FF_RETRIEVAL_CASCADE",
    ),
    "FF_AUTO_WEIGHTS": FeatureFlag(
        name="FF_AUTO_WEIGHTS",
        description="Automatic hybrid weight optimization",
        default=True,
        risk_level=RiskLevel.MEDIUM,
        env_var="FF_AUTO_WEIGHTS",
    ),
    
    # Phase 3 - Ingestion
    "FF_CONTEXTUAL_ENRICHMENT": FeatureFlag(
        name="FF_CONTEXTUAL_ENRICHMENT",
        description="Add document context to chunks during ingestion",
        default=True,
        risk_level=RiskLevel.LOW,
        env_var="FF_CONTEXTUAL_ENRICHMENT",
    ),
    "FF_MULTI_GRANULARITY": FeatureFlag(
        name="FF_MULTI_GRANULARITY",
        description="Index at multiple granularity levels",
        default=False,
        risk_level=RiskLevel.MEDIUM,
        env_var="FF_MULTI_GRANULARITY",
    ),
    "FF_STRUCTURED_DATA": FeatureFlag(
        name="FF_STRUCTURED_DATA",
        description="Parse JSON/CSV/YAML for retrieval",
        default=True,
        risk_level=RiskLevel.LOW,
        env_var="FF_STRUCTURED_DATA",
    ),
    "FF_VISION_INDEX": FeatureFlag(
        name="FF_VISION_INDEX",
        description="Vision-native PDF vectorization (ColPali)",
        default=False,
        risk_level=RiskLevel.HIGH,
        env_var="FF_VISION_INDEX",
    ),
    
    # Phase 4 - Observability & Security
    "FF_CIRCUIT_BREAKERS": FeatureFlag(
        name="FF_CIRCUIT_BREAKERS",
        description="Enable circuit breakers for external services",
        default=True,
        risk_level=RiskLevel.LOW,
        env_var="FF_CIRCUIT_BREAKERS",
    ),
    "FF_COST_TRACKING": FeatureFlag(
        name="FF_COST_TRACKING",
        description="Track and report API costs",
        default=True,
        risk_level=RiskLevel.LOW,
        env_var="FF_COST_TRACKING",
    ),
    
    # High Risk - Requires explicit enablement
    "FF_TOOL_USE": FeatureFlag(
        name="FF_TOOL_USE",
        description="Allow LLM tool execution",
        default=False,
        risk_level=RiskLevel.CRITICAL,
        env_var="FF_TOOL_USE",
    ),
}


class FeatureFlagRegistry:
    """Central registry for managing feature flags.
    
    Checks flags in order of priority:
    1. Runtime overrides (set via API)
    2. Environment variables
    3. Config file settings
    4. Default values
    """
    
    def __init__(self, flags: dict[str, FeatureFlag] | None = None) -> None:
        """Initialize registry with feature definitions."""
        self._flags = flags or V2_FEATURE_FLAGS.copy()
        self._overrides: dict[str, bool] = {}
        self._listeners: list[Callable[[str, bool], None]] = []

    def is_enabled(self, flag_name: str, context: dict[str, Any] | None = None) -> bool:
        """Check if a feature flag is enabled.
        
        Args:
            flag_name: Name of the feature flag
            context: Optional context for percentage-based flags
            
        Returns:
            True if enabled, False otherwise
        """
        flag = self._flags.get(flag_name)
        if not flag:
            return False
        
        # Check runtime override first
        if flag_name in self._overrides:
            return self._overrides[flag_name]
        
        # Check environment variable
        if flag.env_var:
            env_value = os.getenv(flag.env_var, "").lower()
            if env_value in ("true", "1", "yes", "on"):
                return True
            if env_value in ("false", "0", "no", "off"):
                return False
        
        # Check dependencies
        for dep in flag.dependencies:
            if not self.is_enabled(dep, context):
                return False
        
        # Return default
        return flag.default

    def enable(self, flag_name: str) -> None:
        """Enable a feature flag at runtime."""
        if flag_name in self._flags:
            self._overrides[flag_name] = True
            self._notify_listeners(flag_name, True)

    def disable(self, flag_name: str) -> None:
        """Disable a feature flag at runtime."""
        if flag_name in self._flags:
            self._overrides[flag_name] = False
            self._notify_listeners(flag_name, False)

    def reset(self, flag_name: str) -> None:
        """Reset flag to default (remove override)."""
        if flag_name in self._overrides:
            del self._overrides[flag_name]

    def reset_all(self) -> None:
        """Reset all flags to defaults."""
        self._overrides.clear()

    def add_listener(self, listener: Callable[[str, bool], None]) -> None:
        """Add a listener for flag changes."""
        self._listeners.append(listener)

    def _notify_listeners(self, flag_name: str, value: bool) -> None:
        """Notify listeners of flag change."""
        for listener in self._listeners:
            try:
                listener(flag_name, value)
            except Exception:
                pass

    def get_flag(self, flag_name: str) -> FeatureFlag | None:
        """Get flag definition."""
        return self._flags.get(flag_name)

    def list_flags(self) -> list[str]:
        """List all registered flag names."""
        return list(self._flags.keys())

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Get current state of all flags."""
        return {
            name: {
                "enabled": self.is_enabled(name),
                "default": flag.default,
                "risk_level": flag.risk_level.value,
                "description": flag.description,
                "overridden": name in self._overrides,
            }
            for name, flag in self._flags.items()
        }

    def get_flags_by_risk(self, risk_level: RiskLevel) -> list[str]:
        """Get flags filtered by risk level."""
        return [
            name for name, flag in self._flags.items()
            if flag.risk_level == risk_level
        ]


# Global registry singleton
_registry: FeatureFlagRegistry | None = None


def get_feature_flags() -> FeatureFlagRegistry:
    """Get the global feature flag registry."""
    global _registry
    if _registry is None:
        _registry = FeatureFlagRegistry()
    return _registry


def is_feature_enabled(flag_name: str) -> bool:
    """Shortcut to check if a feature is enabled."""
    return get_feature_flags().is_enabled(flag_name)


__all__ = [
    "FeatureState",
    "RiskLevel",
    "FeatureFlag",
    "FeatureFlagRegistry",
    "V2_FEATURE_FLAGS",
    "get_feature_flags",
    "is_feature_enabled",
]
