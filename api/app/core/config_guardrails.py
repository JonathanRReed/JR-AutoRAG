"""Configuration validation and guardrails.

This module implements P1.10: Configuration Guardrails
- Disable graph toggles when "Not Built"
- Warn on conflicting config
- Validate before apply
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("autorag.config_guardrails")


class ValidationLevel(str, Enum):
    """Severity of validation issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single configuration validation issue."""

    level: ValidationLevel
    field: str
    message: str
    suggestion: str | None = None

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "field": self.field,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    disabled_features: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ValidationLevel.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == ValidationLevel.WARNING)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.error_count,
            "warnings": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "disabled_features": self.disabled_features,
        }


class ConfigGuardrails:
    """Validate configuration and enforce guardrails.

    Checks:
    1. GraphRAG toggle disabled when graph not built
    2. RAPTOR toggle disabled when tree not built
    3. Warns on conflicting settings
    4. Validates budget/timeout ranges
    """

    def __init__(
        self,
        graph_built: bool = False,
        raptor_built: bool = False,
    ) -> None:
        self.graph_built = graph_built
        self.raptor_built = raptor_built

    def validate(self, config: dict[str, Any]) -> ValidationResult:
        """Validate configuration settings.

        Args:
            config: Configuration dictionary

        Returns:
            ValidationResult with issues and disabled features
        """
        issues = []
        disabled = []

        # Check 1: GraphRAG availability
        use_graph = config.get("use_graph", False) or config.get("enable_graph_rag", False)
        if use_graph and not self.graph_built:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="use_graph",
                message="GraphRAG enabled but graph not built",
                suggestion="Build graph first or disable GraphRAG",
            ))
            disabled.append("graph_rag")

        # Check 2: RAPTOR availability
        use_raptor = config.get("use_raptor", False) or config.get("enable_raptor", False)
        if use_raptor and not self.raptor_built:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="use_raptor",
                message="RAPTOR enabled but tree not built",
                suggestion="Build RAPTOR tree first or disable",
            ))
            disabled.append("raptor")

        # Check 3: Compression vs retrieval budget conflict
        compression_enabled = config.get("compression_enabled", True)
        retrieval_budget = config.get("retrieval_token_budget", 0) or config.get("max_context_tokens", 0)

        if not compression_enabled and retrieval_budget > 4000:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="compression_enabled",
                message="Compression disabled but retrieval budget is high",
                suggestion="Enable compression to avoid context overflow",
            ))

        # Check 4: Timeout ranges
        timeout = config.get("query_timeout_secs", 30)
        if timeout < 5:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="query_timeout_secs",
                message=f"Timeout very short ({timeout}s)",
                suggestion="Increase timeout to at least 10 seconds",
            ))
        elif timeout > 300:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="query_timeout_secs",
                message=f"Timeout very long ({timeout}s)",
                suggestion="Consider shorter timeout for better UX",
            ))

        # Check 5: Token budgets
        answer_budget = config.get("answer_token_budget", 0)
        if answer_budget > 0 and answer_budget < 100:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                field="answer_token_budget",
                message=f"Answer budget very low ({answer_budget} tokens)",
                suggestion="Increase to at least 200 for useful answers",
            ))

        # Check 6: Grounded mode without corpus
        query_mode = config.get("query_mode", "grounded")
        corpus_empty = config.get("corpus_doc_count", 0) == 0
        if query_mode == "grounded" and corpus_empty:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                field="query_mode",
                message="Grounded mode with no documents",
                suggestion="Add documents or switch to open_domain mode",
            ))

        # Determine if valid (no errors)
        valid = all(i.level != ValidationLevel.ERROR for i in issues)

        return ValidationResult(
            valid=valid,
            issues=issues,
            disabled_features=disabled,
        )

    def get_disabled_toggles(self) -> dict[str, dict]:
        """Get list of toggles that should be disabled.

        Returns map of toggle name to reason.
        """
        disabled = {}

        if not self.graph_built:
            disabled["use_graph"] = {
                "reason": "Graph not built",
                "cta": "Build GraphRAG",
                "action": "build_graph",
            }

        if not self.raptor_built:
            disabled["use_raptor"] = {
                "reason": "RAPTOR tree not built",
                "cta": "Build RAPTOR",
                "action": "build_raptor",
            }

        return disabled


def validate_config(
    config: dict[str, Any],
    graph_built: bool = False,
    raptor_built: bool = False,
) -> ValidationResult:
    """Convenience function to validate config."""
    guardrails = ConfigGuardrails(
        graph_built=graph_built,
        raptor_built=raptor_built,
    )
    return guardrails.validate(config)


__all__ = [
    "ValidationLevel",
    "ValidationIssue",
    "ValidationResult",
    "ConfigGuardrails",
    "validate_config",
]
