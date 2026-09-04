from __future__ import annotations

import socket
from unittest.mock import MagicMock

import pytest

from app.core import providers
from app.core.providers import ProviderError, discover_models, resolve_provider_api_key
from app.schemas.config import DeploymentProfile, ProviderConfig, is_public_provider_url


@pytest.mark.asyncio
async def test_local_provider_discovery_rejects_private_network_url() -> None:
    cfg = ProviderConfig(name="lm", base_url="http://10.0.0.5:8080")

    with pytest.raises(ProviderError, match="localhost or loopback"):
        await discover_models(cfg)


@pytest.mark.parametrize(
    "url",
    [
        "http://100.64.0.1",
        "http://100.127.255.254",
        "http://169.254.1.1",
        "http://192.168.1.10",
        "http://224.0.0.1",
        "http://239.255.255.250",
        "http://[ff02::1]",
        "http://[64:ff9b::1]",
    ],
)
def test_public_provider_url_rejects_non_global_addresses(url: str) -> None:
    assert is_public_provider_url(url) is False


def test_public_provider_url_rejects_dns_multicast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_getaddrinfo(
        *args: object, **kwargs: object
    ) -> list[tuple[object, object, object, object, tuple[str, int]]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("239.255.255.250", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    assert is_public_provider_url("https://models.example.test") is False


def test_public_provider_url_allows_global_address() -> None:
    assert is_public_provider_url("https://8.8.8.8") is True


def test_custom_host_never_inherits_standard_provider_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = MagicMock()
    vault.get.side_effect = lambda key: (
        "openai-secret" if key == "OPENAI_API_KEY" else None
    )
    monkeypatch.setattr(providers, "get_secrets_vault", lambda: vault)

    resolved = resolve_provider_api_key("OpenAI", "https://attacker.example/v1")

    assert resolved is None
    assert vault.get.call_args.args[0] != "OPENAI_API_KEY"


@pytest.mark.asyncio
async def test_model_discovery_rejects_untrusted_public_cloud_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ProviderConfig(
        name="OpenAI",
        base_url="https://models.example.test/v1",
        api_key="do-not-forward",
    )
    monkeypatch.setattr(providers, "is_public_provider_url", lambda _url: True)

    with pytest.raises(ProviderError, match="trusted cloud provider"):
        await discover_models(cfg)


@pytest.mark.asyncio
async def test_local_only_policy_blocks_trusted_cloud_model_discovery() -> None:
    cfg = ProviderConfig(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        api_key="do-not-forward",
    )

    with pytest.raises(ProviderError, match="Local-only mode"):
        await discover_models(cfg, deployment_profile=DeploymentProfile.LOCAL_ONLY)


def test_official_openai_origin_can_use_standard_environment_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault = MagicMock()
    vault.get.side_effect = lambda key: (
        "openai-secret" if key == "OPENAI_API_KEY" else None
    )
    monkeypatch.setattr(providers, "get_secrets_vault", lambda: vault)

    resolved = resolve_provider_api_key("OpenAI", "https://api.openai.com/v1")

    assert resolved == "openai-secret"
    vault.get.assert_called_once_with("OPENAI_API_KEY")
