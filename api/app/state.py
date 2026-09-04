from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .core.orchestrator import Orchestrator

from .schemas.config import AppConfig

_config = AppConfig()
_orchestrator: Optional["Orchestrator"] = None


def get_config() -> AppConfig:
    return _config


def set_config(cfg: AppConfig) -> None:
    global _config
    _config = cfg


def get_orchestrator() -> Optional["Orchestrator"]:
    """Get the global orchestrator instance."""
    return _orchestrator


def set_orchestrator(orch: "Orchestrator") -> None:
    """Set the global orchestrator instance."""
    global _orchestrator
    _orchestrator = orch
