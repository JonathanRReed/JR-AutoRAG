"""Local-first backend registry and policy enforcement."""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas.config import AppConfig, BackendConfig, DeploymentProfile, FallbackConfig


class LocalFirstPolicyError(ValueError):
    """Raised when a configuration or runtime selection violates local-first policy."""


@dataclass(frozen=True)
class BackendResolution:
    subsystem: str
    primary: BackendConfig
    fallback: FallbackConfig

    @property
    def fallback_ids(self) -> list[str]:
        return list(self.fallback.order)


class LocalFirstRegistry:
    """Resolves configured backends and validates local-only constraints."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._backends = dict(config.backends)
        self._fallbacks = dict(config.fallbacks)

    @property
    def config(self) -> AppConfig:
        return self._config

    def refresh(self, config: AppConfig) -> None:
        self._config = config
        self._backends = dict(config.backends)
        self._fallbacks = dict(config.fallbacks)

    def get_backend(self, subsystem: str) -> BackendConfig:
        backend = self._backends.get(subsystem)
        if backend is None:
            raise LocalFirstPolicyError(f"No backend configured for subsystem '{subsystem}'.")
        return backend

    def get_fallback(self, subsystem: str) -> FallbackConfig:
        return self._fallbacks.get(subsystem, FallbackConfig())

    def resolve(self, subsystem: str) -> BackendResolution:
        return BackendResolution(
            subsystem=subsystem,
            primary=self.get_backend(subsystem),
            fallback=self.get_fallback(subsystem),
        )

    def ensure_runtime_allowed(self, subsystem: str) -> BackendConfig:
        backend = self.get_backend(subsystem)
        if self._config.deployment_profile == DeploymentProfile.LOCAL_ONLY:
            if backend.capabilities.requires_network or backend.capabilities.mode.value != "local":
                raise LocalFirstPolicyError(
                    f"Subsystem '{subsystem}' cannot use backend '{backend.backend_id}' in local-only mode."
                )
        return backend

    def describe(self) -> dict[str, object]:
        return {
            "deployment_profile": self._config.deployment_profile.value,
            "backends": {
                key: backend.model_dump()
                for key, backend in self._backends.items()
            },
            "fallbacks": {
                key: fallback.model_dump()
                for key, fallback in self._fallbacks.items()
            },
        }
